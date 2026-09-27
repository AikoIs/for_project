from __future__ import annotations
import copy
import sys
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import argparse
import duckdb
import hashlib
import json
import numpy as np
import polars as pl
import time
import warnings

PREDICTION_HOUR = 6.0



def common_cohort(cohort: pl.DataFrame, prevalent_by_label: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, list[dict]]:
    df = cohort.select('stay_id', 'epoch')
    cols = []
    for label, prevalent in prevalent_by_label.items():
        col = f'_prev_{label}'
        flags = prevalent.select('stay_id').unique().with_columns(**{col: pl.lit(True)})
        df = df.join(flags, on='stay_id', how='left').with_columns(**{col: pl.col(col).fill_null(False)})
        cols.append(col)
    df = df.with_columns(retained=~pl.any_horizontal(cols)) if cols else df.with_columns(retained=pl.lit(True))
    summary = []
    for _key, g in sorted(df.group_by('epoch'), key=lambda kv: kv[0]):
        rec = {'epoch': g['epoch'][0], 'n_stays': g.height, 'n_common_cohort': int(g['retained'].sum()), 'frac_retained': round(float(g['retained'].mean()), 4)}
        for label in prevalent_by_label:
            rec[f'lost_to_{label}'] = int(g[f'_prev_{label}'].sum())
        summary.append(rec)
    return (df.filter('retained').select('stay_id', 'epoch'), summary)

def prevalent_from_onsets(onsets: pl.DataFrame, prediction_hour: float=PREDICTION_HOUR) -> pl.DataFrame:
    return onsets.filter(pl.col('onset_hours') <= prediction_hour).select('stay_id')


TUT = Path(__file__).resolve().parent
OUTPUT = TUT / "output"
PROJECT_ROOT = OUTPUT
CONFIG_ROOT = TUT / "configs"

MIMIC_IV_CSV = r"C:/path/to/mimic-iv-3.1"
EICU_CSV = r"C:/path/to/eicu-crd-2.0"

CONFIGS = {'concepts/eicu.yaml': {'labs': {'creatinine': {'names': ['creatinine'], 'valid': [0.1, 50]}, 'bilirubin_total': {'names': ['total bilirubin'], 'valid': [0.1, 100]}, 'platelets': {'names': ['platelets x 1000'], 'valid': [1, 2000]}, 'pao2': {'names': ['pao2'], 'valid': [10, 800]}, 'lactate': {'names': ['lactate'], 'valid': [0.1, 40]}, 'wbc': {'names': ['wbc x 1000'], 'valid': [0.1, 200]}, 'bun': {'names': ['bun'], 'valid': [1, 300]}, 'sodium': {'names': ['sodium'], 'valid': [90, 200]}, 'potassium': {'names': ['potassium'], 'valid': [1, 12]}, 'bicarbonate': {'names': ['bicarbonate', 'hco3'], 'valid': [5, 60]}, 'hemoglobin': {'names': ['hgb'], 'valid': [2, 25]}, 'glucose': {'names': ['glucose'], 'valid': [10, 2000]}}, 'vitals': {'heart_rate': {'column': 'heartrate', 'valid': [10, 300]}, 'sbp': {'column': 'systemicsystolic', 'valid': [20, 300]}, 'dbp': {'column': 'systemicdiastolic', 'valid': [5, 200]}, 'map': {'column': 'systemicmean', 'valid': [10, 250]}, 'resp_rate': {'column': 'respiration', 'valid': [2, 80]}, 'spo2': {'column': 'sao2', 'valid': [20, 100]}, 'temperature': {'column': 'temperature', 'valid': [25, 45]}}, 'vitals_aperiodic': {'sbp': {'column': 'noninvasivesystolic', 'valid': [20, 300]}, 'dbp': {'column': 'noninvasivediastolic', 'valid': [5, 200]}, 'map': {'column': 'noninvasivemean', 'valid': [10, 250]}}, 'gcs': {'total_valname': 'GCS Total', 'component_valnames': ['Eyes', 'Motor', 'Verbal'], 'valid': [3, 15]}, 'output': {'urine': {'like': ['%urine%', '%foley%', '%void%'], 'exclude_like': ['%irrigant%'], 'valid': [0, 5000]}}, 'weight': {'column': 'admissionweight', 'valid': [20, 400]}, 'infusion': {'norepinephrine': ['%norepinephrine%', '%levophed%'], 'epinephrine': ['%epinephrine%'], 'dopamine': ['%dopamine%'], 'dobutamine': ['%dobutamine%']}, 'ventilation': {'source': 'respiratoryCharting'}, 'culture': {'source': 'treatment', 'like': ['%cultures%']}, 'antibiotics': {'source': 'treatment', 'like': ['%antibacterial%', '%antibiotic%']}}, 'concepts/mimic_iv.yaml': {'labs': {'creatinine': {'itemids': [50912, 52546], 'unit': 'mg/dL', 'valid': [0.1, 50]}, 'bilirubin_total': {'itemids': [50885, 53089], 'unit': 'mg/dL', 'valid': [0.1, 100]}, 'platelets': {'itemids': [51265, 53189], 'unit': 'K/uL', 'valid': [1, 2000]}, 'pao2': {'itemids': [50821], 'unit': 'mmHg', 'valid': [10, 800]}, 'lactate': {'itemids': [50813], 'unit': 'mmol/L', 'valid': [0.1, 40]}, 'wbc': {'itemids': [51301, 51755], 'unit': 'K/uL', 'valid': [0.1, 200]}, 'bun': {'itemids': [51006], 'unit': 'mg/dL', 'valid': [1, 300]}, 'sodium': {'itemids': [50983, 52623], 'unit': 'mEq/L', 'valid': [90, 200]}, 'potassium': {'itemids': [50971, 52610], 'unit': 'mEq/L', 'valid': [1, 12]}, 'bicarbonate': {'itemids': [50882, 50803], 'unit': 'mEq/L', 'valid': [5, 60]}, 'hemoglobin': {'itemids': [51222, 50811], 'unit': 'g/dL', 'valid': [2, 25]}, 'glucose': {'itemids': [50931, 50809], 'unit': 'mg/dL', 'valid': [10, 2000]}}, 'chart': {'heart_rate': {'itemids': [220045], 'valid': [10, 300]}, 'sbp': {'itemids': [220050, 220179], 'valid': [20, 300]}, 'dbp': {'itemids': [220051, 220180], 'valid': [5, 200]}, 'map': {'itemids': [220052, 220181, 225312], 'valid': [10, 250]}, 'resp_rate': {'itemids': [220210, 224690], 'valid': [2, 80]}, 'temperature_c': {'itemids': [223762], 'valid': [25, 45]}, 'temperature_f': {'itemids': [223761], 'valid': [77, 113]}, 'spo2': {'itemids': [220277], 'valid': [20, 100]}, 'fio2': {'itemids': [223835], 'valid': [0.2, 100]}, 'gcs_eye': {'itemids': [220739], 'valid': [1, 4]}, 'gcs_verbal': {'itemids': [223900], 'valid': [1, 5]}, 'gcs_motor': {'itemids': [223901], 'valid': [1, 6]}, 'weight_kg': {'itemids': [226512, 224639], 'valid': [20, 400]}, 'weight_lbs': {'itemids': [226531], 'valid': [44, 880]}, 'height_cm': {'itemids': [226730], 'valid': [100, 250]}}, 'output': {'urine_output': {'itemids': [226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227489], 'subtract_itemids': [227488], 'valid': [0, 5000]}}, 'input': {'norepinephrine': {'itemids': [221906]}, 'epinephrine': {'itemids': [221289, 229617]}, 'dopamine': {'itemids': [221662]}, 'dobutamine': {'itemids': [221653]}, 'vasopressin': {'itemids': [222315]}, 'phenylephrine': {'itemids': [221749, 229630, 229632]}}, 'procedure': {'invasive_ventilation': {'itemids': [225792]}, 'noninvasive_ventilation': {'itemids': [225794]}}, 'culture': {'source': 'microbiologyevents', 'require_positive': False, 'exclude_spec_type_patterns': ['%SCREEN%', '%MRSA%', '%VRE%', '%SURVEILLANCE%'], 'exclude_spec_types': ['STAPH AUREUS SWAB'], 'sterile_sites': ['BLOOD CULTURE', 'BLOOD CULTURE - NEONATE', 'BLOOD CULTURE (POST-MORTEM)', 'FLUID RECEIVED IN BLOOD CULTURE BOTTLES', 'CSF;SPINAL FLUID', 'PERITONEAL FLUID', 'PLEURAL FLUID', 'PERICARDIAL FLUID', 'JOINT FLUID', 'DIALYSIS FLUID', 'BILE', 'ABSCESS', 'TISSUE', 'BONE MARROW']}, 'antibiotics': {'routes_excluded': ['OU', 'OS', 'OD', 'AU', 'AS', 'AD', 'IRR', 'TP', 'BOTH EYES', 'LEFT EYE', 'RIGHT EYE', 'DESENSITIZATION', 'EAR', 'EYE'], 'names': ['adoxa', 'ala-tet', 'alodox', 'amikacin', 'amikin', 'amoxicillin', 'amphotericin', 'ampicillin', 'augmentin', 'avelox', 'avidoxy', 'azactam', 'azithromycin', 'aztreonam', 'bactocill', 'bactrim', 'bethkis', 'biaxin', 'bicillin', 'cayston', 'cefazolin', 'cefepime', 'cefotan', 'cefotetan', 'cefotaxime', 'cefpodoxime', 'cefoxitin', 'ceftazidime', 'ceftin', 'ceftriaxone', 'cefuroxime', 'cephalexin', 'chloramphenicol', 'cipro', 'ciprofloxacin', 'claforan', 'clarithromycin', 'cleocin', 'clindamycin', 'cubicin', 'dalvance', 'daptomycin', 'declomycin', 'dicloxacillin', 'dirithromycin', 'doryx', 'doxycycline', 'duricef', 'dynacin', 'ery-tab', 'eryped', 'eryc', 'erythrocin', 'erythromycin', 'factive', 'flagyl', 'fortaz', 'furadantin', 'garamycin', 'gentamicin', 'kanamycin', 'keflex', 'ketek', 'levaquin', 'levofloxacin', 'lincocin', 'linezolid', 'macrobid', 'macrodantin', 'maxipime', 'mefoxin', 'meropenem', 'methicillin', 'metronidazole', 'minocin', 'minocycline', 'monodox', 'monurol', 'morgidox', 'moxatag', 'moxifloxacin', 'myrac', 'nafcillin', 'neomycin', 'nicazel', 'nitrofurantoin', 'norfloxacin', 'ocudox', 'ofloxacin', 'omnicef', 'oracea', 'oraxyl', 'oxacillin', 'pc pen vk', 'penicillin', 'periostat', 'pfizerpen', 'piperacillin', 'primsol', 'proquin', 'raniclor', 'rifadin', 'rifampin', 'rocephin', 'smz-tmp', 'septra', 'solodyn', 'spectracef', 'streptomycin', 'sulfadiazine', 'sulfamethoxazole', 'sulfatrim', 'sulfisoxazole', 'suprax', 'synercid', 'tazicef', 'tetracycline', 'timentin', 'tinidazole', 'tobi', 'tobramycin', 'trimethoprim', 'unasyn', 'vancocin', 'vancomycin', 'vantin', 'vibativ', 'vibra-tabs', 'vibramycin', 'zithromax', 'zosyn', 'zyvox']}}, 'datasets/conversion.yaml': {'defaults': {'sample_size': 10000000, 'compression': 'zstd', 'row_group_size': 1000000}, 'sources': {'mimic_iv': {'tables': {'hosp/patients': {}, 'hosp/admissions': {}, 'hosp/transfers': {}, 'hosp/services': {}, 'hosp/provider': {}, 'hosp/d_labitems': {}, 'hosp/d_icd_diagnoses': {}, 'hosp/d_icd_procedures': {}, 'hosp/d_hcpcs': {}, 'hosp/diagnoses_icd': {'types': {'icd_code': 'VARCHAR', 'icd_version': 'BIGINT'}}, 'hosp/procedures_icd': {'types': {'icd_code': 'VARCHAR', 'icd_version': 'BIGINT'}}, 'hosp/hcpcsevents': {}, 'hosp/drgcodes': {}, 'hosp/omr': {'types': {'result_value': 'VARCHAR'}}, 'hosp/labevents': {'types': {'value': 'VARCHAR', 'valuenum': 'DOUBLE', 'valueuom': 'VARCHAR', 'flag': 'VARCHAR', 'priority': 'VARCHAR', 'comments': 'VARCHAR', 'ref_range_lower': 'DOUBLE', 'ref_range_upper': 'DOUBLE', 'order_provider_id': 'VARCHAR'}}, 'hosp/microbiologyevents': {'types': {'comments': 'VARCHAR', 'quantity': 'VARCHAR', 'dilution_text': 'VARCHAR', 'dilution_comparison': 'VARCHAR', 'dilution_value': 'DOUBLE', 'interpretation': 'VARCHAR', 'ab_name': 'VARCHAR', 'org_name': 'VARCHAR'}}, 'hosp/prescriptions': {'types': {'gsn': 'VARCHAR', 'ndc': 'VARCHAR', 'formulary_drug_cd': 'VARCHAR', 'dose_val_rx': 'VARCHAR', 'form_val_disp': 'VARCHAR', 'doses_per_24_hrs': 'DOUBLE', 'poe_id': 'VARCHAR', 'drug': 'VARCHAR'}}, 'hosp/emar': {'types': {'medication': 'VARCHAR', 'event_txt': 'VARCHAR', 'scheduletime': 'TIMESTAMP', 'storetime': 'TIMESTAMP'}}, 'hosp/emar_detail': {'all_varchar': True}, 'hosp/pharmacy': {'all_varchar': True}, 'hosp/poe': {'all_varchar': True}, 'icu/icustays': {}, 'icu/d_items': {}, 'icu/caregiver': {}, 'icu/chartevents': {'types': {'value': 'VARCHAR', 'valuenum': 'DOUBLE', 'valueuom': 'VARCHAR', 'warning': 'BIGINT'}}, 'icu/outputevents': {'types': {'value': 'DOUBLE', 'valueuom': 'VARCHAR'}}, 'icu/inputevents': {'types': {'amountuom': 'VARCHAR', 'rateuom': 'VARCHAR', 'totalamountuom': 'VARCHAR', 'ordercategoryname': 'VARCHAR', 'secondaryordercategoryname': 'VARCHAR', 'ordercomponenttypedescription': 'VARCHAR', 'ordercategorydescription': 'VARCHAR', 'originalamount': 'DOUBLE', 'originalrate': 'DOUBLE', 'patientweight': 'DOUBLE'}}, 'icu/procedureevents': {'types': {'value': 'DOUBLE', 'valueuom': 'VARCHAR', 'location': 'VARCHAR', 'locationcategory': 'VARCHAR', 'ordercategoryname': 'VARCHAR', 'ordercategorydescription': 'VARCHAR', 'secondaryordercategoryname': 'VARCHAR', 'originalamount': 'DOUBLE', 'originalrate': 'DOUBLE'}}, 'icu/datetimeevents': {'types': {'value': 'VARCHAR', 'valueuom': 'VARCHAR'}}, 'icu/ingredientevents': {'types': {'amountuom': 'VARCHAR', 'rateuom': 'VARCHAR', 'originalamount': 'DOUBLE', 'originalrate': 'DOUBLE'}}}}, 'eicu': {'tables': {'patient': {}, 'hospital': {}, 'admissionDx': {}, 'apacheApsVar': {}, 'apachePatientResult': {}, 'apachePredVar': {}, 'diagnosis': {'types': {'icd9code': 'VARCHAR', 'diagnosisstring': 'VARCHAR'}}, 'treatment': {'types': {'treatmentstring': 'VARCHAR'}}, 'lab': {'types': {'labresult': 'DOUBLE', 'labresulttext': 'VARCHAR', 'labmeasurenamesystem': 'VARCHAR', 'labmeasurenameinterface': 'VARCHAR'}}, 'microLab': {}, 'medication': {'all_varchar': True}, 'admissionDrug': {'all_varchar': True}, 'infusionDrug': {'all_varchar': True}, 'intakeOutput': {'types': {'cellvaluenumeric': 'DOUBLE', 'cellvaluetext': 'VARCHAR', 'celllabel': 'VARCHAR', 'cellpath': 'VARCHAR'}}, 'vitalPeriodic': {}, 'vitalAperiodic': {}, 'nurseCharting': {'types': {'nursingchartvalue': 'VARCHAR', 'nursingchartcelltypevalname': 'VARCHAR'}}, 'respiratoryCare': {'all_varchar': True}, 'respiratoryCharting': {'all_varchar': True}, 'carePlanGeneral': {'all_varchar': True}, 'carePlanGoal': {'all_varchar': True}, 'carePlanEOL': {'all_varchar': True}, 'carePlanCareProvider': {'all_varchar': True}, 'carePlanInfectiousDisease': {'all_varchar': True}, 'pastHistory': {'all_varchar': True}, 'allergy': {'all_varchar': True}, 'customLab': {'all_varchar': True}, 'physicalExam': {'skip': True, 'reason': 'not used; large and unstructured'}, 'nurseAssessment': {'skip': True, 'reason': 'not used; large and unstructured'}, 'nurseCare': {'skip': True, 'reason': 'not used; large and unstructured'}, 'note': {'skip': True, 'reason': 'free-text notes are not used in this study'}}}}}, 'experiments/eicu.yaml': {'designs': {'hospital_cv': {'n_folds': 3, 'seed': 20260920}, 'leave_one_region_out': {'regions': ['Midwest', 'South', 'West', 'Northeast', 'Unknown']}}, 'bootstrap_unit': 'hospital', 'contrasts_primary': [{'name': 'DiD_aki', 'label_a': 'aki_urine', 'label_b': 'aki_creatinine'}], 'contrasts_secondary': [{'name': 'DiD_sepsis', 'label_a': 'sepsis3', 'label_b': 'sofa_dysfunction'}], 'training_sizes': ['full', 'matched'], 'training_sizes_secondary': ['event_matched'], 'models': ['logreg', 'xgboost', 'gru'], 'n_seeds': 15, 'n_folds': 5, 'bootstrap': {'n_boot': 2000, 'levels': [0.95, 0.9], 'resample_patients': True, 'resample_seeds': True, 'seed': 20260920, 'decompose_n_boot': 600}, 'sesoi': 0.01, 'multiplicity': {'method': 'holm', 'family_size': 12, 'alpha': 0.05}, 'equivalence_may_be_unattainable': True}, 'experiments/main.yaml': {'train_epochs': ['2008 - 2010', '2011 - 2013'], 'target_epochs': ['2014 - 2016', '2017 - 2019', '2020 - 2022'], 'models': ['logreg', 'xgboost', 'gru'], 'training_sizes': ['full', 'matched'], 'n_seeds': 15, 'n_folds': 5, 'bootstrap': {'n_boot': 2000, 'levels': [0.95, 0.9], 'resample_patients': True, 'resample_seeds': True, 'seed': 20260920, 'decompose_n_boot': 600}, 'contrasts': [{'name': 'DiD_sepsis', 'label_a': 'sepsis3', 'label_b': 'sofa_dysfunction'}, {'name': 'DiD_aki', 'label_a': 'aki_urine', 'label_b': 'aki_creatinine'}], 'multiplicity': {'method': 'holm', 'family_size': 12, 'alpha': 0.05}, 'sesoi': 0.01}, 'experiments/power.yaml': {'placeholder': True, 'n_bootstrap': 500, 'n_replicates': 3, 'seed': 20260919, 'sesoi': 0.01, 'contrasts': [{'name': 'DiD_sepsis', 'label_a': {'name': 'sepsis3', 'prevalence': 0.08, 'oracle_auroc': 0.75}, 'label_b': {'name': 'sofa_dysfunction', 'prevalence': 0.18, 'oracle_auroc': 0.72}, 'latent_corr': 0.8}, {'name': 'DiD_aki', 'label_a': {'name': 'aki_urine', 'prevalence': 0.2, 'oracle_auroc': 0.74}, 'label_b': {'name': 'aki_creatinine', 'prevalence': 0.15, 'oracle_auroc': 0.78}, 'latent_corr': 0.55}], 'epoch_sizes': [6000, 9000, 12000, 15000, 20000], 'holdout_fraction': 0.3, 'gap_b': 0.03, 'true_did': [0.0, 0.005, 0.01, 0.02, 0.03, 0.05], 'n_models': 3, 'n_seeds': 5, 'model_skill_sd': 0.01, 'seed_skill_sd': 0.004, 'n_folds': 5, 'fold_skill_sd': 0.008}, 'experiments/power_real.yaml': {'placeholder': False, 'n_bootstrap': 500, 'n_replicates': 3, 'seed': 20260919, 'sesoi': 0.01, 'contrasts': [{'name': 'DiD_sepsis', 'label_a': {'name': 'sepsis3', 'prevalence': 0.1275, 'oracle_auroc': 0.75}, 'label_b': {'name': 'sofa_dysfunction', 'prevalence': 0.2876, 'oracle_auroc': 0.72}, 'latent_corr': 0.8}, {'name': 'DiD_aki', 'label_a': {'name': 'aki_urine', 'prevalence': 0.4204, 'oracle_auroc': 0.74}, 'label_b': {'name': 'aki_creatinine', 'prevalence': 0.1606, 'oracle_auroc': 0.78}, 'latent_corr': 0.55}], 'epoch_sizes': [10551, 9727, 6974], 'holdout_fraction': 0.3, 'gap_b': 0.03, 'true_did': [0.0, 0.005, 0.01, 0.02, 0.03, 0.05], 'n_models': 3, 'n_seeds': 5, 'model_skill_sd': 0.01, 'seed_skill_sd': 0.004, 'n_folds': 5, 'fold_skill_sd': 0.008, '_note': 'epoch_sizes and prevalences measured on the common cohort at the frozen design; oracle_auroc values remain assumptions, as no model has been fitted'}, 'features/physiology_only.yaml': {'window_hours': 6, 'name': 'physiology_only', 'vitals': ['heart_rate', 'sbp', 'dbp', 'map', 'resp_rate', 'spo2', 'temperature', 'gcs_total'], 'labs': ['creatinine', 'platelets', 'bilirubin_total', 'lactate', 'wbc', 'bun', 'sodium', 'potassium', 'bicarbonate', 'hemoglobin', 'glucose', 'pao2'], 'derived': ['urine_rate'], 'static': ['age', 'sex', 'admission_type'], 'aggregations': ['min', 'max', 'mean', 'first', 'last']}, 'labels/primary.yaml': {'task': {'prediction_hour': 6.0, 'horizon_hours': 48.0, 'max_hours': 96.0, 'cohort': 'common'}, 'sofa': {'baseline_rule': 'at_prediction', 'increase': 2, 'prevalence_rule': 'none'}, 'suspicion': {'culture_variant': 'no_surveillance', 'antibiotic': 'all_routes'}, 'labels': {'aki_creatinine': {'rank': 1, 'criterion': 'KDIGO serum creatinine, stage >= 1', 'baseline': 'rolling_min_7d'}, 'sofa_dysfunction': {'rank': 2, 'criterion': 'SOFA >= baseline + 2, no infection criterion'}, 'aki_urine': {'rank': 3, 'criterion': 'KDIGO urine output, stage >= 1', 'max_gap_hours': 4.0, 'min_coverage_ratio': 0.75}, 'sepsis3': {'rank': 4, 'criterion': 'SOFA >= baseline + 2 within [-48 h, +24 h] of suspected infection', 'onset_rule': 'later_of_both', 'abx_then_culture_hours': 24.0, 'culture_then_abx_hours': 72.0}}, 'primary_contrasts': [{'name': 'DiD_sepsis', 'label_a': 'sepsis3', 'label_b': 'sofa_dysfunction'}, {'name': 'DiD_aki', 'label_a': 'aki_urine', 'label_b': 'aki_creatinine'}], 'sensitivity': {'sofa_baseline_rule': ['min_before', 'zero'], 'sofa_prevalence_rule': ['sofa_at_threshold', 'rise_before'], 'prediction_hour': [4.0], 'horizon_hours': [24.0, 72.0], 'sepsis_onset_rule': ['suspicion', 'sofa'], 'culture_variant': ['any', 'sterile_only'], 'antibiotic': ['systemic_only'], 'creatinine_baseline': ['first_of_stay'], 'kdigo_min_stage': [2], 'cohort': ['per_label'], 'drop_epoch': ['2020 - 2022'], 'restrict_epochs_before_culture_break': ['2008 - 2010', '2011 - 2013', '2014 - 2016']}}, 'models/search_space.yaml': {'inner_cv': {'epochs': ['2008 - 2010', '2011 - 2013'], 'n_folds': 5, 'fold_seed': 1, 'model_seed': 0}, 'logreg': {'C': [0.01, 0.1, 1.0, 10.0], 'max_iter': [2000], 'class_weight': [None, 'balanced'], '_grid': ['C', 'class_weight'], '_extra': [{'C': 0.003}, {'C': 30.0}, {'C': 0.03}, {'C': 3.0}]}, 'xgboost': {'max_depth': [3, 4, 6], 'learning_rate': [0.03, 0.1], 'n_estimators': [300, 600], 'min_child_weight': [5.0, 20.0], 'subsample': [0.8], 'colsample_bytree': [0.8], 'device': ['cuda'], '_grid': ['max_depth', 'learning_rate', 'n_estimators'], '_extra': [{'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 800, 'min_child_weight': 20.0}, {'max_depth': 6, 'learning_rate': 0.03, 'n_estimators': 800, 'min_child_weight': 20.0}, {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 300, 'min_child_weight': 20.0}, {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 400, 'min_child_weight': 20.0}, {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 600, 'min_child_weight': 20.0}, {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 500, 'min_child_weight': 10.0}]}, 'gru': {'hidden': [32, 64, 128], 'layers': [1, 2], 'dropout': [0.2], 'lr': [0.001], 'epochs': [40], 'batch_size': [256], 'patience': [6], 'device': ['cuda'], '_grid': ['hidden', 'layers'], '_extra': [{'hidden': 64, 'layers': 2, 'dropout': 0.4}, {'hidden': 128, 'layers': 1, 'lr': 0.0003}]}}, 'models/smoke.yaml': {'logreg': {'C': 1.0, 'max_iter': 2000}, 'xgboost': {'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5.0, 'device': 'cuda'}, 'gru': {'hidden': 64, 'layers': 2, 'dropout': 0.2, 'lr': 0.001, 'epochs': 40, 'batch_size': 256, 'patience': 6, 'device': 'cuda'}}, 'models/tuned.yaml': {'logreg': {'max_iter': 2000, 'C': 10.0, 'class_weight': 'balanced'}, 'xgboost': {'min_child_weight': 5.0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'device': 'cuda', 'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 600}, 'gru': {'dropout': 0.2, 'lr': 0.001, 'epochs': 40, 'batch_size': 256, 'patience': 6, 'device': 'cuda', 'hidden': 64, 'layers': 1}}}

PATHS = {
    "sources": {
        "mimic_iv": {"version": "3.1",
                      "csv_root": MIMIC_IV_CSV},
        "eicu": {"version": "2.0",
                  "csv_root": EICU_CSV},
    },
    "data_root": str(OUTPUT / "data"),
    "parquet_root": str(OUTPUT / "data" / "parquet"),
    "interim_root": str(OUTPUT / "data" / "interim"),
    "processed_root": str(OUTPUT / "data" / "processed"),
    "results_root": str(OUTPUT / "results"),
    "figures_root": str(OUTPUT / "figures"),
    "tables_root": str(OUTPUT / "tables"),
    "duckdb": {'memory_limit': '48GB', 'threads': 24, 'temp_directory': str(OUTPUT / 'data' / '_duckdb_tmp')},
    "min_cell_count": 20,
}


def load_config(relative):
    on_disk = CONFIG_ROOT / relative
    if on_disk.exists():
        return yaml.safe_load(on_disk.read_text(encoding="utf-8"))
    return copy.deepcopy(CONFIGS[relative])


def paths():
    return copy.deepcopy(PATHS)






def parquet_dir(source):
    return Path(PATHS["parquet_root"]) / source



MIN_AGE = 18

MIN_LOS_HOURS = 12.0

def register_source(con: duckdb.DuckDBPyConnection, source: str, tables: list[str]) -> None:
    root: Path = parquet_dir(source)
    for t in tables:
        path = root / f'{t}.parquet'
        if not path.exists():
            raise FileNotFoundError(path)
        con.execute(f"CREATE OR REPLACE VIEW {source}_{t.lower()} AS SELECT * FROM read_parquet('{path.as_posix()}')")

def build_cohort(con: duckdb.DuckDBPyConnection, table: str='cohort') -> None:
    register_source(con, 'mimic_iv', ['patients', 'admissions', 'icustays'])
    con.execute(f'\n        CREATE OR REPLACE TABLE {table} AS\n        WITH ranked AS (\n            SELECT\n                s.subject_id, s.hadm_id, s.stay_id, s.intime, s.outtime,\n                s.los * 24.0 AS los_hours,\n                s.first_careunit,\n                ROW_NUMBER() OVER (PARTITION BY s.subject_id ORDER BY s.intime) AS stay_rank\n            FROM mimic_iv_icustays s\n        )\n        SELECT\n            r.subject_id, r.hadm_id, r.stay_id, r.intime, r.outtime, r.los_hours,\n            r.first_careunit,\n            p.gender,\n            p.anchor_year_group AS epoch,\n            -- age at the admission that contains this ICU stay\n            p.anchor_age + (EXTRACT(year FROM a.admittime) - p.anchor_year) AS age,\n            EXTRACT(year FROM r.intime) - p.anchor_year AS anchor_offset_years,\n            a.admittime, a.dischtime, a.admission_type, a.race, a.insurance,\n            a.hospital_expire_flag\n        FROM ranked r\n        JOIN mimic_iv_patients p USING (subject_id)\n        JOIN mimic_iv_admissions a USING (hadm_id)\n        WHERE r.stay_rank = 1\n          AND r.los_hours >= {MIN_LOS_HOURS}\n          AND p.anchor_age + (EXTRACT(year FROM a.admittime) - p.anchor_year) >= {MIN_AGE}\n        ')



VENT_SETTING_ITEMIDS = [220339, 224685, 224684, 224687, 224738, 223848, 224688]

VENT_MODE_ITEMIDS = [223849, 229314]

GCS_EYE, GCS_VERBAL, GCS_MOTOR = (220739, 223900, 223901)

_RATE_SQL = "\n    CASE lower(i.rateuom)\n        WHEN 'mcg/kg/min' THEN i.rate\n        WHEN 'mg/kg/min'  THEN i.rate * 1000.0\n        WHEN 'mcg/min'    THEN i.rate / nullif(w.weight_kg, 0)\n        WHEN 'mg/min'     THEN i.rate * 1000.0 / nullif(w.weight_kg, 0)\n        WHEN 'mcg/kg/hr'  THEN i.rate / 60.0\n        WHEN 'mg/kg/hr'   THEN i.rate * 1000.0 / 60.0\n        ELSE NULL\n    END\n"

_HOURS = "date_diff('second', c.intime, {t}) / 3600.0"

def _cfg() -> dict:
    return load_config('concepts/mimic_iv.yaml')

def lab(con: duckdb.DuckDBPyConnection, concept: str, max_hours: float=96.0, lookback_days: int=7) -> pl.DataFrame:
    spec = _cfg()['labs'][concept]
    ids = ', '.join(map(str, spec['itemids']))
    lo, hi = spec['valid']
    return con.execute(f"\n        SELECT c.stay_id, {_HOURS.format(t='l.charttime')} AS hours, l.valuenum AS value\n        FROM mimic_iv_labevents l\n        JOIN cohort c ON c.hadm_id = l.hadm_id\n        WHERE l.itemid IN ({ids}) AND l.valuenum BETWEEN {lo} AND {hi}\n          AND l.charttime BETWEEN c.intime - INTERVAL {lookback_days} DAY\n                              AND c.intime + INTERVAL {int(max_hours)} HOUR\n        ").pl()

def chart(con: duckdb.DuckDBPyConnection, concept: str, max_hours: float=96.0) -> pl.DataFrame:
    spec = _cfg()['chart'][concept]
    ids = ', '.join(map(str, spec['itemids']))
    lo, hi = spec['valid']
    return con.execute(f"\n        SELECT e.stay_id, {_HOURS.format(t='e.charttime')} AS hours, e.valuenum AS value\n        FROM mimic_iv_chartevents e\n        JOIN cohort c ON c.stay_id = e.stay_id\n        WHERE e.itemid IN ({ids}) AND e.valuenum BETWEEN {lo} AND {hi}\n          AND e.charttime >= c.intime AND e.charttime <= c.intime + INTERVAL {int(max_hours)} HOUR\n        ").pl()

def gcs_total(con: duckdb.DuckDBPyConnection, max_hours: float=96.0) -> pl.DataFrame:
    return con.execute(f"\n        WITH parts AS (\n            SELECT e.stay_id, e.charttime,\n                   max(CASE WHEN e.itemid = {GCS_EYE}    THEN e.valuenum END) AS eye,\n                   max(CASE WHEN e.itemid = {GCS_VERBAL} THEN e.valuenum END) AS verbal,\n                   max(CASE WHEN e.itemid = {GCS_MOTOR}  THEN e.valuenum END) AS motor,\n                   any_value(c.intime) AS intime\n            FROM mimic_iv_chartevents e\n            JOIN cohort c ON c.stay_id = e.stay_id\n            WHERE e.itemid IN ({GCS_EYE}, {GCS_VERBAL}, {GCS_MOTOR})\n              AND e.valuenum IS NOT NULL\n              AND e.charttime >= c.intime\n              AND e.charttime <= c.intime + INTERVAL {int(max_hours)} HOUR\n            GROUP BY e.stay_id, e.charttime\n        )\n        SELECT stay_id,\n               date_diff('second', intime, charttime) / 3600.0 AS hours,\n               eye + verbal + motor AS value\n        FROM parts\n        WHERE eye IS NOT NULL AND verbal IS NOT NULL AND motor IS NOT NULL\n        ").pl()

def ventilated_hours(con: duckdb.DuckDBPyConnection, max_hours: float=96.0) -> pl.DataFrame:
    ids = ', '.join(map(str, VENT_SETTING_ITEMIDS + VENT_MODE_ITEMIDS))
    return con.execute(f"\n        SELECT DISTINCT e.stay_id, {_HOURS.format(t='e.charttime')} AS hours\n        FROM mimic_iv_chartevents e\n        JOIN cohort c ON c.stay_id = e.stay_id\n        WHERE e.itemid IN ({ids})\n          AND (e.valuenum IS NOT NULL OR e.value IS NOT NULL)\n          AND e.charttime >= c.intime AND e.charttime <= c.intime + INTERVAL {int(max_hours)} HOUR\n        ").pl()

def urine_output(con: duckdb.DuckDBPyConnection, max_hours: float=96.0) -> pl.DataFrame:
    spec = _cfg()['output']['urine_output']
    add = ', '.join(map(str, spec['itemids']))
    sub = ', '.join(map(str, spec.get('subtract_itemids', []))) or '-1'
    return con.execute(f"\n        SELECT o.stay_id, {_HOURS.format(t='o.charttime')} AS hours,\n               sum(CASE WHEN o.itemid IN ({sub}) THEN -o.value ELSE o.value END) AS value\n        FROM mimic_iv_outputevents o\n        JOIN cohort c ON c.stay_id = o.stay_id\n        WHERE o.itemid IN ({add}, {sub}) AND o.value BETWEEN 0 AND 5000\n          AND o.charttime >= c.intime AND o.charttime <= c.intime + INTERVAL {int(max_hours)} HOUR\n        GROUP BY 1, 2\n        ").pl()

def weights(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    return con.execute('\n        WITH w AS (\n            SELECT c.stay_id, c.gender,\n                   first(CASE WHEN e.itemid = 226531 THEN e.valuenum * 0.45359237\n                              ELSE e.valuenum END ORDER BY e.charttime) AS weight_kg\n            FROM mimic_iv_chartevents e\n            JOIN cohort c ON c.stay_id = e.stay_id\n            WHERE e.itemid IN (226512, 224639, 226531)\n              AND e.valuenum IS NOT NULL\n              AND e.charttime <= c.intime + INTERVAL 48 HOUR\n            GROUP BY c.stay_id, c.gender\n            HAVING first(CASE WHEN e.itemid = 226531 THEN e.valuenum * 0.45359237\n                              ELSE e.valuenum END ORDER BY e.charttime) BETWEEN 20 AND 400\n        ), med AS (\n            SELECT gender, median(weight_kg) AS fallback FROM w GROUP BY gender\n        )\n        SELECT c.stay_id,\n               coalesce(w.weight_kg, med.fallback) AS weight_kg,\n               w.weight_kg IS NULL AS weight_imputed\n        FROM cohort c\n        LEFT JOIN w USING (stay_id)\n        LEFT JOIN med ON med.gender = c.gender\n        ').pl()

def vasopressor_rates(con: duckdb.DuckDBPyConnection, max_hours: float=96.0) -> pl.DataFrame:
    cfg = _cfg()['input']
    drugs = {d: cfg[d]['itemids'] for d in ('norepinephrine', 'epinephrine', 'dopamine', 'dobutamine')}
    cases = ' '.join((f"WHEN i.itemid IN ({', '.join(map(str, ids))}) THEN '{name}'" for name, ids in drugs.items()))
    all_ids = ', '.join((str(i) for ids in drugs.values() for i in ids))
    return con.execute(f"\n        WITH w AS (SELECT stay_id, weight_kg FROM weights_tmp),\n        infusions AS (\n            SELECT i.stay_id,\n                   CASE {cases} END AS drug,\n                   {_RATE_SQL} AS rate,\n                   {_HOURS.format(t='i.starttime')} AS start_h,\n                   {_HOURS.format(t='i.endtime')}   AS end_h\n            FROM mimic_iv_inputevents i\n            JOIN cohort c ON c.stay_id = i.stay_id\n            LEFT JOIN w ON w.stay_id = i.stay_id\n            WHERE i.itemid IN ({all_ids}) AND i.rate IS NOT NULL AND i.rate > 0\n              AND i.endtime >= c.intime\n              AND i.starttime <= c.intime + INTERVAL {int(max_hours)} HOUR\n        )\n        SELECT stay_id, drug, rate,\n               CAST(h AS DOUBLE) AS hours\n        FROM infusions,\n             LATERAL generate_series(\n                 greatest(CAST(floor(start_h) AS BIGINT), 0),\n                 least(CAST(ceil(end_h) AS BIGINT), {int(max_hours)})\n             ) AS g(h)\n        WHERE rate IS NOT NULL\n        ").pl()

def antibiotic_times(con: duckdb.DuckDBPyConnection, lookback_hours: float=24.0, max_hours: float=96.0, systemic_only: bool=False) -> pl.DataFrame:
    cfg = _cfg()['antibiotics']
    likes = ' OR '.join((f"lower(p.drug) LIKE '%{n}%'" for n in cfg['names']))
    routes = ', '.join((f"'{r}'" for r in cfg['routes_excluded']))
    systemic = "AND upper(p.route) IN ('IV','IV DRIP','IVPCA','IM','IV BOLUS')" if systemic_only else ''
    return con.execute(f"\n        SELECT DISTINCT c.stay_id, {_HOURS.format(t='p.starttime')} AS hours\n        FROM mimic_iv_prescriptions p\n        JOIN cohort c ON c.hadm_id = p.hadm_id\n        WHERE ({likes})\n          AND (p.route IS NULL OR upper(p.route) NOT IN ({routes}))\n          {systemic}\n          AND p.starttime BETWEEN c.intime - INTERVAL {int(lookback_hours)} HOUR\n                              AND c.intime + INTERVAL {int(max_hours)} HOUR\n        ").pl()

def culture_times(con: duckdb.DuckDBPyConnection, lookback_hours: float=24.0, max_hours: float=96.0, variant: str='no_surveillance') -> pl.DataFrame:
    cfg = _cfg()['culture']
    if variant == 'any':
        where = '1=1'
    elif variant == 'sterile_only':
        sites = ', '.join((f"'{s}'" for s in cfg['sterile_sites']))
        where = f'upper(m.spec_type_desc) IN ({sites})'
    elif variant == 'no_surveillance':
        clauses = [f"upper(m.spec_type_desc) NOT LIKE '{p}'" for p in cfg['exclude_spec_type_patterns']]
        clauses += [f"upper(m.spec_type_desc) <> '{s}'" for s in cfg['exclude_spec_types']]
        where = ' AND '.join(clauses)
    else:
        raise ValueError(f'unknown culture variant {variant!r}')
    return con.execute(f"\n        SELECT DISTINCT c.stay_id,\n               date_diff('second', c.intime,\n                         coalesce(m.charttime, m.chartdate)) / 3600.0 AS hours\n        FROM mimic_iv_microbiologyevents m\n        JOIN cohort c ON c.hadm_id = m.hadm_id\n        WHERE {where}\n          AND coalesce(m.charttime, m.chartdate)\n              BETWEEN c.intime - INTERVAL {int(lookback_hours)} HOUR\n                  AND c.intime + INTERVAL {int(max_hours)} HOUR\n        ").pl()

STATIC_NUMERIC = ('age',)

STATIC_CATEGORICAL = ('sex', 'admission_type')

def FeatureMatrix(stay_ids, tabular, tabular_columns, sequence, sequence_columns, static, static_columns):
    n = len(stay_ids)
    if not len(tabular) == len(sequence) == len(static) == n:
        raise ValueError('views are not aligned')
    return {'stay_ids': stay_ids, 'tabular': tabular, 'tabular_columns': tabular_columns, 'sequence': sequence, 'sequence_columns': sequence_columns, 'static': static, 'static_columns': static_columns}

def fm_n_stays(fm):
    return len(fm['stay_ids'])

def fm_subset(fm, mask):
    return FeatureMatrix(stay_ids=fm['stay_ids'][mask], tabular=fm['tabular'][mask], tabular_columns=fm['tabular_columns'], sequence=fm['sequence'][mask], sequence_columns=fm['sequence_columns'], static=fm['static'][mask], static_columns=fm['static_columns'])

def hourly_panel(events: dict[str, pl.DataFrame], stay_ids: pl.Series, window_hours: int) -> pl.DataFrame:
    grid = pl.DataFrame({'stay_id': stay_ids}).join(pl.DataFrame({'hour': list(range(window_hours))}), how='cross')
    panel = grid
    for name, df in events.items():
        if df.height == 0:
            panel = panel.with_columns(**{name: pl.lit(None, pl.Float64)})
            continue
        agg = pl.col('value').sum() if name == 'urine_rate' else pl.col('value').mean()
        binned = df.filter((pl.col('hours') >= 0) & (pl.col('hours') < window_hours)).with_columns(hour=pl.col('hours').floor().cast(pl.Int64)).group_by(['stay_id', 'hour']).agg(**{name: agg})
        panel = panel.join(binned, on=['stay_id', 'hour'], how='left')
    return panel.sort(['stay_id', 'hour'])

def build(panel: pl.DataFrame, static: pl.DataFrame, variables: list[str], aggregations: list[str], window_hours: int) -> FeatureMatrix:
    panel = panel.sort(['stay_id', 'hour'])
    stay_ids = panel['stay_id'].unique(maintain_order=True)
    n, t = (stay_ids.len(), window_hours)
    filled = panel.with_columns([pl.col(v).forward_fill().over('stay_id') for v in variables])
    seq = filled.select(variables).to_numpy().astype(np.float64).reshape(n, t, len(variables))
    exprs = []
    for v in variables:
        for a in aggregations:
            col = pl.col(v)
            e = {'min': col.min(), 'max': col.max(), 'mean': col.mean(), 'first': col.drop_nulls().first(), 'last': col.drop_nulls().last()}[a]
            exprs.append(e.alias(f'{v}__{a}'))
    tab_df = panel.group_by('stay_id', maintain_order=True).agg(exprs)
    tab_df = pl.DataFrame({'stay_id': stay_ids}).join(tab_df, on='stay_id', how='left')
    tabular_columns = [c for c in tab_df.columns if c != 'stay_id']
    tabular = tab_df.select(tabular_columns).to_numpy().astype(np.float64)
    static_df = pl.DataFrame({'stay_id': stay_ids}).join(static, on='stay_id', how='left')
    static_cols: list[str] = []
    static_parts: list[np.ndarray] = []
    for c in STATIC_NUMERIC:
        static_parts.append(static_df[c].to_numpy().astype(np.float64)[:, None])
        static_cols.append(c)
    for c in STATIC_CATEGORICAL:
        levels = sorted((x for x in static_df[c].unique().to_list() if x is not None))
        for lv in levels:
            static_parts.append((static_df[c] == lv).to_numpy().astype(np.float64)[:, None])
            static_cols.append(f'{c}={lv}')
    static_arr = np.hstack(static_parts) if static_parts else np.zeros((n, 0))
    return FeatureMatrix(stay_ids=stay_ids.to_numpy(), tabular=np.hstack([tabular, static_arr]), tabular_columns=tabular_columns + static_cols, sequence=seq, sequence_columns=list(variables), static=static_arr, static_columns=static_cols)

def assert_no_future_leakage(events: dict[str, pl.DataFrame], window_hours: int) -> None:
    offenders = {name: int((df['hours'] >= window_hours).sum()) for name, df in events.items() if df.height and 'hours' in df.columns}
    bad = {k: v for k, v in offenders.items() if v}
    if bad:
        raise AssertionError(f'events at or after hour {window_hours}: {bad}')

LABELS = ('aki_creatinine', 'sofa_dysfunction', 'aki_urine', 'sepsis3')

def dataset_load(path):
    z = np.load(Path(path), allow_pickle=False)
    return {'stay_ids': z['stay_ids'], 'tabular': z['tabular'], 'sequence': z['sequence'], 'static': z['static'], 'epochs': z['epochs'].astype(str), 'folds': z['folds'], 'tabular_columns': list(z['tabular_columns'].astype(str)), 'sequence_columns': list(z['sequence_columns'].astype(str)), 'static_columns': list(z['static_columns'].astype(str)), 'labels': {k: z['y_' + k] for k in LABELS if 'y_' + k in z}}

def dataset_matrix(ds):
    return FeatureMatrix(stay_ids=ds['stay_ids'], tabular=ds['tabular'], tabular_columns=ds['tabular_columns'], sequence=ds['sequence'], sequence_columns=ds['sequence_columns'], static=ds['static'], static_columns=ds['static_columns'])

def dataset_y(ds, label):
    if label not in ds['labels']:
        raise KeyError('unknown label ' + repr(label) + '; have ' + str(sorted(ds['labels'])))
    return ds['labels'][label]

def dataset_epoch_mask(ds, *epochs):
    return np.isin(ds['epochs'], np.array(epochs))

def preprocessor_fit(fm, fitted_on):
    tab = fm['tabular']
    tab_median = _nanmedian(tab, axis=0)
    tab_filled = _fill(tab, tab_median)
    tab_mean = tab_filled.mean(axis=0)
    tab_std = _safe_std(tab_filled)
    seq = fm['sequence']
    flat = seq.reshape(-1, seq.shape[-1])
    seq_median = _nanmedian(flat, axis=0)
    flat_filled = _fill(flat, seq_median)
    seq_mean = flat_filled.mean(axis=0)
    seq_std = _safe_std(flat_filled)
    return {'tab_median': tab_median, 'tab_mean': tab_mean, 'tab_std': tab_std, 'seq_median': seq_median, 'seq_mean': seq_mean, 'seq_std': seq_std, 'fitted_on': fitted_on, 'n_fit': fm_n_stays(fm)}

def preprocessor_transform(pre, fm):
    tab = (_fill(fm['tabular'], pre['tab_median']) - pre['tab_mean']) / pre['tab_std']
    seq = fm['sequence']
    flat = _fill(seq.reshape(-1, seq.shape[-1]), pre['seq_median'])
    flat = (flat - pre['seq_mean']) / pre['seq_std']
    return (tab, flat.reshape(seq.shape))

def preprocessor_describe(pre):
    return {'fitted_on': pre['fitted_on'], 'n_fit': pre['n_fit'], 'n_tabular': int(pre['tab_median'].size), 'n_sequence_vars': int(pre['seq_median'].size)}

def _nanmedian(x: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=axis)
    return np.where(np.isfinite(med), med, 0.0)

def _fill(x: np.ndarray, median: np.ndarray) -> np.ndarray:
    out = np.array(x, dtype=np.float64, copy=True)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.broadcast_to(median, out.shape)[bad]
    return out

def _safe_std(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0)
    return np.where(sd > 1e-12, sd, 1.0)


def connect(read_only_views: bool=False) -> duckdb.DuckDBPyConnection:
    cfg = paths()['duckdb']
    tmp = Path(cfg['temp_directory'])
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg['memory_limit']}'")
    con.execute(f"SET threads={int(cfg['threads'])}")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute('SET preserve_insertion_order=false')
    if read_only_views:
        register_views(con)
    return con

def register_views(con: duckdb.DuckDBPyConnection) -> None:
    root = Path(paths()['parquet_root'])
    if not root.exists():
        return
    for source_dir in sorted(root.iterdir()):
        if not source_dir.is_dir():
            continue
        for pq in sorted(source_dir.rglob('*.parquet')):
            view = f'{source_dir.name}_{pq.stem.lower()}'
            con.execute(f"""CREATE OR REPLACE VIEW "{view}" AS SELECT * FROM read_parquet('{pq.as_posix()}')""")

_RATIO_STAGE_1 = 1.5

_RATIO_STAGE_2 = 2.0

_RATIO_STAGE_3 = 3.0

_ABSOLUTE_RISE_48H = 0.3

_ABSOLUTE_STAGE_3 = 4.0

def CreatinineBaseline(method='rolling_min_7d', window='7d'):
    return {'method': method, 'window': window}

def creatinine_stage(creatinine: pl.DataFrame, baseline=None) -> pl.DataFrame:
    if baseline is None:
        baseline = CreatinineBaseline()
    _require(creatinine, ['stay_id', 'hours', 'value'])
    df = creatinine.drop_nulls('value').sort(['stay_id', 'hours'])
    if df.height == 0:
        return df.with_columns(base_7d=pl.lit(None, pl.Float64), min_48h=pl.lit(None, pl.Float64), stage=pl.lit(None, pl.Int8))
    df = df.with_columns(_t=_as_datetime(pl.col('hours')))
    if baseline['method'] == 'rolling_min_7d':
        base = pl.col('value').rolling_min_by('_t', baseline['window'], closed='right')
    elif baseline['method'] == 'first_of_stay':
        base = pl.col('value').filter(pl.col('hours') >= 0).first()
    elif baseline['method'] == 'mdrd_75':
        raise NotImplementedError("mdrd_75 needs age/sex/race and is built in the concept layer, not here; pass the resulting baseline as 'first_of_stay' input")
    else:
        raise ValueError(f"unknown baseline method {baseline['method']!r}")
    df = df.with_columns(base_7d=base.over('stay_id'), min_48h=pl.col('value').rolling_min_by('_t', '48h', closed='right').over('stay_id'))
    ratio = pl.col('value') / pl.col('base_7d')
    rose_0_3 = pl.col('value') >= pl.col('min_48h') + _ABSOLUTE_RISE_48H
    stage = pl.when((ratio >= _RATIO_STAGE_3) | (pl.col('value') >= _ABSOLUTE_STAGE_3)).then(3).when(ratio >= _RATIO_STAGE_2).then(2).when((ratio >= _RATIO_STAGE_1) | rose_0_3).then(1).otherwise(0).cast(pl.Int8)
    return df.with_columns(stage=stage).drop('_t')

def urine_output_stage(urine: pl.DataFrame, weights: pl.DataFrame, min_coverage_ratio: float=0.75, max_gap_hours: float=4.0) -> pl.DataFrame:
    _require(urine, ['stay_id', 'hours', 'value'])
    _require(weights, ['stay_id', 'weight_kg'])
    df = urine.drop_nulls('value').filter(pl.col('value') >= 0).sort(['stay_id', 'hours']).join(weights, on='stay_id', how='left')
    if df.height == 0:
        return df.with_columns(stage=pl.lit(None, pl.Int8))
    df = df.with_columns(_t=_as_datetime(pl.col('hours')), _row=pl.int_range(pl.len(), dtype=pl.UInt32), _one=pl.lit(1.0))
    anchors = df.select('stay_id', _anchor=pl.col('hours')).sort('_anchor')
    for window_h in (6, 12, 24):
        span = f'{window_h}h'
        df = df.with_columns(**{f'_sum_{window_h}': pl.col('value').rolling_sum_by('_t', span, closed='right').over('stay_id'), f'_n_{window_h}': pl.col('_one').rolling_sum_by('_t', span, closed='right').over('stay_id')})
        probe = df.select('_row', 'stay_id', _target=pl.col('hours') - window_h).sort('_target')
        prev = probe.join_asof(anchors, left_on='_target', right_on='_anchor', by='stay_id', strategy='backward').select('_row', **{f'_prev_{window_h}': pl.col('_anchor').fill_null(0.0)})
        df = df.join(prev, on='_row', how='left')
        covered = pl.col('hours') - pl.col(f'_prev_{window_h}')
        rate = pl.col(f'_sum_{window_h}') / covered / pl.col('weight_kg')
        dense_enough = pl.col(f'_n_{window_h}') >= window_h / max_gap_hours
        df = df.with_columns(**{f'rate_{window_h}h': pl.when((covered >= min_coverage_ratio * window_h) & (covered <= window_h / min_coverage_ratio) & dense_enough).then(rate).otherwise(None)})
    df = df.sort(['stay_id', 'hours'])
    evaluable = pl.col('rate_6h').is_not_null() | pl.col('rate_12h').is_not_null() | pl.col('rate_24h').is_not_null()
    stage = pl.when(~evaluable).then(None).when((pl.col('rate_24h') < 0.3) | (pl.col('rate_12h') <= 0.0)).then(3).when(pl.col('rate_12h') < 0.5).then(2).when(pl.col('rate_6h') < 0.5).then(1).otherwise(0).cast(pl.Int8)
    drop = [c for c in df.columns if c.startswith('_')]
    return df.with_columns(evaluable=evaluable, stage=stage).drop(drop)

def first_onset(staged: pl.DataFrame, min_stage: int=1) -> pl.DataFrame:
    _require(staged, ['stay_id', 'hours', 'stage'])
    return staged.filter(pl.col('stage') >= min_stage).sort(['stay_id', 'hours']).group_by('stay_id').agg(onset_hours=pl.col('hours').first(), onset_stage=pl.col('stage').first()).sort('stay_id')

_EPOCH = datetime(2000, 1, 1)

def _as_datetime(hours: pl.Expr) -> pl.Expr:
    ms = (hours * 3600000.0).round(0).cast(pl.Int64).cast(pl.Duration('ms'))
    return pl.lit(_EPOCH, dtype=pl.Datetime('ms')) + ms

def _require(df: pl.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f'missing required columns {missing}; got {df.columns}')

WINDOW_LAB = 24

WINDOW_VITAL = 4

WINDOW_PRESSOR = 1

WINDOW_URINE = 24

_POS_INF = float('inf')

_NEG_INF = float('-inf')

def SofaInputs(pao2, fio2, ventilated, platelets, bilirubin, mean_arterial_pressure, gcs_total, creatinine, urine, vasopressors):
    return {'pao2': pao2, 'fio2': fio2, 'ventilated': ventilated, 'platelets': platelets, 'bilirubin': bilirubin, 'mean_arterial_pressure': mean_arterial_pressure, 'gcs_total': gcs_total, 'creatinine': creatinine, 'urine': urine, 'vasopressors': vasopressors}

def hourly_grid(stays: pl.DataFrame) -> pl.DataFrame:
    _require(stays, ['stay_id', 'max_hours'])
    return stays.with_columns(hour=pl.int_ranges(0, pl.col('max_hours').ceil().cast(pl.Int32) + 1)).explode('hour').select('stay_id', 'hour')

def _rolling_worst(events: pl.DataFrame, grid: pl.DataFrame, window: int, worst: str, name: str) -> pl.DataFrame:
    if worst not in ('min', 'max'):
        raise ValueError(worst)
    sentinel = _POS_INF if worst == 'min' else _NEG_INF
    if events.height == 0:
        return grid.with_columns(**{name: pl.lit(None, pl.Float64)})
    per_hour = events.drop_nulls('value').with_columns(hour=pl.col('hours').floor().cast(pl.Int32)).group_by(['stay_id', 'hour']).agg(v=getattr(pl.col('value'), worst)())
    out = grid.join(per_hour, on=['stay_id', 'hour'], how='left').sort(['stay_id', 'hour']).with_columns(_v=pl.col('v').fill_null(sentinel))
    rolled = getattr(pl.col('_v'), f'rolling_{worst}')(window_size=window, min_samples=1).over('stay_id')
    return out.with_columns(_r=rolled).with_columns(**{name: pl.when(pl.col('_r').is_infinite()).then(None).otherwise(pl.col('_r'))}).drop(['v', '_v', '_r'])

def clean_fio2(value: pl.Expr) -> pl.Expr:
    return pl.when(value >= 20.0).then(value / 100.0).when((value >= 0.2) & (value <= 1.0)).then(value).otherwise(None)

def pf_ratio_events(pao2: pl.DataFrame, fio2: pl.DataFrame, tolerance_hours: float=4.0) -> pl.DataFrame:
    _require(pao2, ['stay_id', 'hours', 'value'])
    _require(fio2, ['stay_id', 'hours', 'value'])
    if pao2.height == 0:
        return pao2.with_columns(value=pl.lit(None, pl.Float64))
    f = fio2.with_columns(value=clean_fio2(pl.col('value'))).drop_nulls('value').select('stay_id', _fh=pl.col('hours'), _fio2=pl.col('value')).sort('_fh')
    left = pao2.drop_nulls('value').sort('hours')
    joined = left.join_asof(f, left_on='hours', right_on='_fh', by='stay_id', strategy='backward', tolerance=tolerance_hours).with_columns(_fio2=pl.col('_fio2').fill_null(0.21))
    return joined.select('stay_id', 'hours', value=pl.col('value') / pl.col('_fio2'))

def hourly_sofa(inputs: SofaInputs, stays: pl.DataFrame) -> pl.DataFrame:
    grid = hourly_grid(stays)
    pf = pf_ratio_events(inputs['pao2'], inputs['fio2'])
    g = _rolling_worst(pf, grid, WINDOW_LAB, 'min', 'pf_ratio')
    vent_hours = inputs['ventilated'].with_columns(hour=pl.col('hours').floor().cast(pl.Int32)).select('stay_id', 'hour').unique().with_columns(_vent=pl.lit(1.0))
    g = g.join(vent_hours, on=['stay_id', 'hour'], how='left').sort(['stay_id', 'hour']).with_columns(ventilated=pl.col('_vent').fill_null(0.0).rolling_max(window_size=WINDOW_LAB, min_samples=1).over('stay_id') > 0).drop('_vent')
    for frame, window, worst, name in ((inputs['platelets'], WINDOW_LAB, 'min', 'platelets'), (inputs['bilirubin'], WINDOW_LAB, 'max', 'bilirubin'), (inputs['creatinine'], WINDOW_LAB, 'max', 'creatinine'), (inputs['mean_arterial_pressure'], WINDOW_VITAL, 'min', 'map'), (inputs['gcs_total'], WINDOW_VITAL, 'min', 'gcs')):
        g = g.join(_rolling_worst(frame, grid, window, worst, name), on=['stay_id', 'hour'], how='left')
    for drug in ('norepinephrine', 'epinephrine', 'dopamine', 'dobutamine'):
        sub = inputs['vasopressors'].filter(pl.col('drug') == drug).select('stay_id', 'hours', value=pl.col('rate'))
        g = g.join(_rolling_worst(sub, grid, WINDOW_PRESSOR, 'max', f'rate_{drug}'), on=['stay_id', 'hour'], how='left')
    if inputs['urine'].height:
        uo = inputs['urine'].drop_nulls('value').with_columns(hour=pl.col('hours').floor().cast(pl.Int32)).group_by(['stay_id', 'hour']).agg(v=pl.col('value').sum())
        g = g.join(uo, on=['stay_id', 'hour'], how='left').sort(['stay_id', 'hour']).with_columns(urine_24h=pl.col('v').fill_null(0.0).rolling_sum(window_size=WINDOW_URINE, min_samples=WINDOW_URINE).over('stay_id'), _charted=pl.col('v').is_not_null().cast(pl.Int8).rolling_sum(window_size=WINDOW_URINE, min_samples=1).over('stay_id')).with_columns(urine_24h=pl.when(pl.col('_charted') >= WINDOW_URINE / 4).then(pl.col('urine_24h')).otherwise(None)).drop(['v', '_charted'])
    else:
        g = g.with_columns(urine_24h=pl.lit(None, pl.Float64))
    return g.with_columns(**_component_scores()).with_columns(sofa_total=sum((pl.col(c).fill_null(0) for c in ('sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal'))).cast(pl.Int8), sofa_components_missing=sum((pl.col(c).is_null().cast(pl.Int8) for c in ('sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal'))).cast(pl.Int8))

def _component_scores() -> dict[str, pl.Expr]:
    pf, vent = (pl.col('pf_ratio'), pl.col('ventilated'))
    resp = pl.when(pf.is_null()).then(None).when((pf < 100) & vent).then(4).when((pf < 200) & vent).then(3).when(pf < 300).then(2).when(pf < 400).then(1).otherwise(0)
    plt = pl.col('platelets')
    coag = pl.when(plt.is_null()).then(None).when(plt < 20).then(4).when(plt < 50).then(3).when(plt < 100).then(2).when(plt < 150).then(1).otherwise(0)
    bili = pl.col('bilirubin')
    liver = pl.when(bili.is_null()).then(None).when(bili >= 12.0).then(4).when(bili >= 6.0).then(3).when(bili >= 2.0).then(2).when(bili >= 1.2).then(1).otherwise(0)
    ne, ep = (pl.col('rate_norepinephrine'), pl.col('rate_epinephrine'))
    dop, dob = (pl.col('rate_dopamine'), pl.col('rate_dobutamine'))
    mapp = pl.col('map')
    any_pressor = ne.is_not_null() | ep.is_not_null() | dop.is_not_null() | dob.is_not_null()
    cardio = pl.when(mapp.is_null() & ~any_pressor).then(None).when((dop > 15) | (ne > 0.1) | (ep > 0.1)).then(4).when((dop > 5) | (ne > 0) | (ep > 0)).then(3).when((dop > 0) | (dob > 0)).then(2).when(mapp < 70).then(1).otherwise(0)
    gcs = pl.col('gcs')
    cns = pl.when(gcs.is_null()).then(None).when(gcs < 6).then(4).when(gcs < 10).then(3).when(gcs < 13).then(2).when(gcs < 15).then(1).otherwise(0)
    cr, uo = (pl.col('creatinine'), pl.col('urine_24h'))
    renal_cr = pl.when(cr.is_null()).then(None).when(cr >= 5.0).then(4).when(cr >= 3.5).then(3).when(cr >= 2.0).then(2).when(cr >= 1.2).then(1).otherwise(0)
    renal_uo = pl.when(uo.is_null()).then(None).when(uo < 200).then(4).when(uo < 500).then(3).otherwise(0)
    renal = pl.when(renal_cr.is_null() & renal_uo.is_null()).then(None).otherwise(pl.max_horizontal(renal_cr.fill_null(0), renal_uo.fill_null(0)))
    return {'sofa_resp': resp.cast(pl.Int8), 'sofa_coag': coag.cast(pl.Int8), 'sofa_liver': liver.cast(pl.Int8), 'sofa_cardio': cardio.cast(pl.Int8), 'sofa_cns': cns.cast(pl.Int8), 'sofa_renal': renal.cast(pl.Int8)}

BASELINE_RULES = ('min_before', 'at_prediction', 'zero')

def baseline_sofa(sofa: pl.DataFrame, prediction_hour: float=6.0, rule: str='at_prediction') -> pl.DataFrame:
    _require(sofa, ['stay_id', 'hour', 'sofa_total'])
    if rule not in BASELINE_RULES:
        raise ValueError(f'baseline rule must be one of {BASELINE_RULES}, got {rule!r}')
    if rule == 'zero':
        return sofa.select('stay_id').unique().with_columns(baseline_sofa=pl.lit(0, pl.Int8))
    if rule == 'min_before':
        return sofa.filter(pl.col('hour') <= prediction_hour).group_by('stay_id').agg(baseline_sofa=pl.col('sofa_total').min())
    return sofa.filter(pl.col('hour') <= prediction_hour).sort(['stay_id', 'hour']).group_by('stay_id').agg(baseline_sofa=pl.col('sofa_total').last())

PREVALENCE_RULES = ('none', 'sofa_at_threshold', 'rise_before')

def organ_dysfunction_onset(sofa: pl.DataFrame, prediction_hour: float=6.0, increase: int=2, baseline_rule: str='at_prediction') -> pl.DataFrame:
    _require(sofa, ['stay_id', 'hour', 'sofa_total'])
    baseline = baseline_sofa(sofa, prediction_hour, baseline_rule)
    return sofa.join(baseline, on='stay_id', how='inner').filter(pl.col('hour') > prediction_hour).filter(pl.col('sofa_total') - pl.col('baseline_sofa') >= increase).sort(['stay_id', 'hour']).group_by('stay_id').agg(onset_hours=pl.col('hour').first().cast(pl.Float64), onset_sofa=pl.col('sofa_total').first(), baseline_sofa=pl.col('baseline_sofa').first()).sort('stay_id')

def prevalent_dysfunction(sofa: pl.DataFrame, prediction_hour: float=6.0, increase: int=2, rule: str='none') -> pl.DataFrame:
    _require(sofa, ['stay_id', 'hour', 'sofa_total'])
    if rule not in PREVALENCE_RULES:
        raise ValueError(f'prevalence rule must be one of {PREVALENCE_RULES}, got {rule!r}')
    if rule == 'none':
        return sofa.select('stay_id').head(0)
    at_t = baseline_sofa(sofa, prediction_hour, 'at_prediction')
    if rule == 'sofa_at_threshold':
        return at_t.filter(pl.col('baseline_sofa') >= increase).select('stay_id')
    min_before = baseline_sofa(sofa, prediction_hour, 'min_before').rename({'baseline_sofa': 'min_sofa'})
    return at_t.join(min_before, on='stay_id', how='inner').filter(pl.col('baseline_sofa') - pl.col('min_sofa') >= increase).select('stay_id')

def _require(df: pl.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f'missing required columns {missing}; got {df.columns}')

ABX_THEN_CULTURE_HOURS = 24.0

CULTURE_THEN_ABX_HOURS = 72.0

SOFA_WINDOW_BEFORE = 48.0

SOFA_WINDOW_AFTER = 24.0

def suspicion_of_infection(antibiotics: pl.DataFrame, cultures: pl.DataFrame, abx_then_culture_hours: float=ABX_THEN_CULTURE_HOURS, culture_then_abx_hours: float=CULTURE_THEN_ABX_HOURS) -> pl.DataFrame:
    _require(antibiotics, ['stay_id', 'hours'])
    _require(cultures, ['stay_id', 'hours'])
    if antibiotics.height == 0 or cultures.height == 0:
        return pl.DataFrame(schema={'stay_id': pl.Int64, 'suspicion_hours': pl.Float64, 'abx_hours': pl.Float64, 'culture_hours': pl.Float64, 'order': pl.String})
    pairs = antibiotics.select('stay_id', abx_hours=pl.col('hours')).join(cultures.select('stay_id', culture_hours=pl.col('hours')), on='stay_id', how='inner')
    delta = pl.col('culture_hours') - pl.col('abx_hours')
    valid = (delta >= 0) & (delta <= abx_then_culture_hours) | (delta < 0) & (-delta <= culture_then_abx_hours)
    return pairs.filter(valid).with_columns(suspicion_hours=pl.min_horizontal('abx_hours', 'culture_hours'), order=pl.when(delta >= 0).then(pl.lit('abx_first')).otherwise(pl.lit('culture_first'))).select('stay_id', 'suspicion_hours', 'abx_hours', 'culture_hours', 'order').sort(['stay_id', 'suspicion_hours'])

def dysfunction_hours(sofa: pl.DataFrame, prediction_hour: float=6.0, increase: int=2, baseline_rule: str='at_prediction', after_prediction_only: bool=True) -> pl.DataFrame:
    baseline = baseline_sofa(sofa, prediction_hour, baseline_rule)
    if after_prediction_only:
        sofa = sofa.filter(pl.col('hour') > prediction_hour)
    return sofa.join(baseline, on='stay_id', how='inner').filter(pl.col('sofa_total') - pl.col('baseline_sofa') >= increase).select('stay_id', 'sofa_total', 'baseline_sofa', dysfunction_hours=pl.col('hour').cast(pl.Float64))

def sepsis3_onset(sofa: pl.DataFrame, suspicion: pl.DataFrame, prediction_hour: float=6.0, increase: int=2, window_before: float=SOFA_WINDOW_BEFORE, window_after: float=SOFA_WINDOW_AFTER, onset_rule: str='later_of_both', baseline_rule: str='at_prediction') -> pl.DataFrame:
    if onset_rule not in ('later_of_both', 'suspicion', 'sofa'):
        raise ValueError(onset_rule)
    dys = dysfunction_hours(sofa, prediction_hour=prediction_hour, increase=increase, baseline_rule=baseline_rule)
    if dys.height == 0 or suspicion.height == 0:
        return pl.DataFrame(schema={'stay_id': pl.Int64, 'onset_hours': pl.Float64, 'suspicion_hours': pl.Float64, 'dysfunction_hours': pl.Float64})
    joined = suspicion.select('stay_id', 'suspicion_hours').unique().join(dys.select('stay_id', 'dysfunction_hours'), on='stay_id', how='inner')
    lag = pl.col('dysfunction_hours') - pl.col('suspicion_hours')
    joined = joined.filter((lag >= -window_before) & (lag <= window_after))
    if joined.height == 0:
        return pl.DataFrame(schema={'stay_id': pl.Int64, 'onset_hours': pl.Float64, 'suspicion_hours': pl.Float64, 'dysfunction_hours': pl.Float64})
    onset = {'later_of_both': pl.max_horizontal('suspicion_hours', 'dysfunction_hours'), 'suspicion': pl.col('suspicion_hours'), 'sofa': pl.col('dysfunction_hours')}[onset_rule]
    return joined.with_columns(onset_hours=onset).filter(pl.col('onset_hours') > prediction_hour).sort(['stay_id', 'onset_hours']).group_by('stay_id').agg(onset_hours=pl.col('onset_hours').first(), suspicion_hours=pl.col('suspicion_hours').first(), dysfunction_hours=pl.col('dysfunction_hours').first()).sort('stay_id')

def prevalent_sepsis3(sofa: pl.DataFrame, suspicion: pl.DataFrame, prediction_hour: float=6.0, increase: int=2, window_before: float=SOFA_WINDOW_BEFORE, window_after: float=SOFA_WINDOW_AFTER, rule: str='none') -> pl.DataFrame:
    if rule == 'none' or suspicion.height == 0:
        return sofa.select('stay_id').head(0)
    dys = dysfunction_hours(sofa, prediction_hour=prediction_hour, increase=increase, baseline_rule='zero' if rule == 'sofa_at_threshold' else 'min_before', after_prediction_only=False).filter(pl.col('dysfunction_hours') <= prediction_hour)
    if dys.height == 0:
        return sofa.select('stay_id').head(0)
    susp = suspicion.select('stay_id', 'suspicion_hours').filter(pl.col('suspicion_hours') <= prediction_hour).unique()
    lag = pl.col('dysfunction_hours') - pl.col('suspicion_hours')
    return susp.join(dys.select('stay_id', 'dysfunction_hours'), on='stay_id', how='inner').filter((lag >= -window_before) & (lag <= window_after)).select('stay_id').unique()

def _require(df: pl.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f'missing required columns {missing}; got {df.columns}')


TARGET_EPOCHS = ('2014 - 2016', '2017 - 2019', '2020 - 2022')

SMOKE_TRAIN_EPOCH = '2008 - 2010'

SMOKE_TEST_EPOCH = '2011 - 2013'

PIPELINE_FROZEN_TAG = 'pipeline-frozen'

def assert_target_epochs_sealed(epochs: tuple[str, ...]) -> None:
    requested = set(epochs) & set(TARGET_EPOCHS)
    if not requested:
        return
    import subprocess
    try:
        tags = subprocess.run(['git', 'tag', '--list', PIPELINE_FROZEN_TAG], capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f'cannot verify the {PIPELINE_FROZEN_TAG} tag: {exc}') from exc
    if PIPELINE_FROZEN_TAG not in tags:
        raise RuntimeError(f'target epochs {sorted(requested)} are sealed until the pipeline is frozen. Tag {PIPELINE_FROZEN_TAG} once features, models and hyperparameters are settled on the {SMOKE_TRAIN_EPOCH} -> {SMOKE_TEST_EPOCH} pair.')

def fold_of(stay_id: int, n_folds: int, seed: int) -> int:
    digest = hashlib.blake2b(f'{seed}:{int(stay_id)}'.encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'big') % n_folds

def assign_folds(stay_ids: np.ndarray, n_folds: int=5, seed: int=0) -> np.ndarray:
    return np.array([fold_of(s, n_folds, seed) for s in stay_ids], dtype=np.int64)


def assert_disjoint(*groups: np.ndarray) -> None:
    seen: set[int] = set()
    for i, g in enumerate(groups):
        ids = {int(x) for x in g}
        overlap = seen & ids
        if overlap:
            raise AssertionError(f'group {i} shares {len(overlap)} stay(s) with an earlier group, e.g. {sorted(overlap)[:3]}')
        seen |= ids

def assert_folds_partition(stay_ids: np.ndarray, folds: np.ndarray, n_folds: int) -> None:
    if len(stay_ids) != len(folds):
        raise AssertionError('fold vector does not match the stay vector')
    present = set(np.unique(folds).tolist())
    if present - set(range(n_folds)):
        raise AssertionError(f'fold ids out of range: {sorted(present)}')
    empty = [k for k in range(n_folds) if not (folds == k).any()]
    if empty:
        raise AssertionError(f'empty folds: {empty}')


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = _midranks(np.asarray(score, dtype=np.float64))
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def _midranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    xs = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    is_new = np.empty(x.size, dtype=bool)
    is_new[0] = True
    np.not_equal(xs[1:], xs[:-1], out=is_new[1:])
    starts = np.flatnonzero(is_new)
    ends = np.append(starts[1:], x.size)
    mid = (starts + ends + 1) / 2.0
    run_id = np.cumsum(is_new) - 1
    ranks[order] = mid[run_id]
    return ranks



MODEL_USES = {'logreg': 'tabular', 'xgboost': 'tabular', 'gru': 'sequence'}

MODEL_DEFAULTS = {
    'logreg': {'C': 1.0, 'max_iter': 2000, 'class_weight': None},
    'xgboost': {'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5.0, 'device': 'cuda'},
    'gru': {'hidden': 64, 'layers': 2, 'dropout': 0.2, 'lr': 0.001, 'epochs': 30, 'batch_size': 256, 'patience': 5, 'device': 'cuda'},
}

def build_model(name, seed=0, **params):
    if name not in MODEL_USES:
        raise ValueError('unknown model ' + repr(name) + '; have ' + str(sorted(MODEL_USES)))
    merged = dict(MODEL_DEFAULTS[name])
    merged.update(params)
    return {'name': name, 'uses': MODEL_USES[name], 'seed': seed, 'params': merged, 'net': None, 'device': None}


def model_select(model, fm, pre):
    tab, seq = preprocessor_transform(pre, fm)
    if model['uses'] == 'tabular':
        return tab
    return seq

def model_fit_matrix(model, fm, y, pre):
    model_fit(model, model_select(model, fm, pre), y)

def model_predict_matrix(model, fm, pre):
    return model_predict_proba(model, model_select(model, fm, pre))

def model_fit(model, x, y):
    if model['name'] == 'logreg':
        fit_logreg(model, x, y)
    elif model['name'] == 'xgboost':
        fit_xgboost(model, x, y)
    else:
        fit_gru(model, x, y)

def model_predict_proba(model, x):
    if model['name'] == 'logreg':
        return model['net'].predict_proba(x)[:, 1]
    if model['name'] == 'xgboost':
        import xgboost as xgb
        return model['net'].predict(xgb.DMatrix(x))
    return predict_gru(model, x)

def fit_logreg(model, x, y):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=model['params']['C'], max_iter=model['params']['max_iter'], class_weight=model['params']['class_weight'], random_state=model['seed'], solver='lbfgs')
    clf.fit(x, y)
    model['net'] = clf

def fit_xgboost(model, x, y):
    import xgboost as xgb
    p = dict(model['params'])
    n_rounds = p.pop('n_estimators')
    model['net'] = xgb.train({**p, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'seed': model['seed'], 'eta': p.pop('learning_rate')}, xgb.DMatrix(x, label=y), num_boost_round=n_rounds)

def build_gru_net(model, n_vars):
    import torch
    from torch import nn
    p = model['params']

    class Net(nn.Module):

        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=n_vars, hidden_size=p['hidden'], num_layers=p['layers'], batch_first=True, dropout=p['dropout'] if p['layers'] > 1 else 0.0)
            self.head = nn.Sequential(nn.Dropout(p['dropout']), nn.Linear(p['hidden'], 1))

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1, :]).squeeze(-1)
    torch.manual_seed(model['seed'])
    return Net()

def fit_gru(model, x, y):
    import torch
    from torch import nn
    p = model['params']
    model['device'] = torch.device(p['device'] if torch.cuda.is_available() and p['device'] == 'cuda' else 'cpu')
    rng = np.random.default_rng(model['seed'])
    idx = rng.permutation(len(y))
    n_val = max(1, int(0.15 * len(y)))
    val_idx, tr_idx = (idx[:n_val], idx[n_val:])
    xb = torch.tensor(x, dtype=torch.float32)
    yb = torch.tensor(y, dtype=torch.float32)
    net = build_gru_net(model, x.shape[-1]).to(model['device'])
    model['net'] = net
    pos = float(yb[tr_idx].sum())
    pos_weight = torch.tensor([(len(tr_idx) - pos) / max(pos, 1.0)], device=model['device'])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=p['lr'])
    best, best_state, waited = (float('inf'), None, 0)
    for _epoch in range(p['epochs']):
        net.train()
        order = torch.tensor(rng.permutation(len(tr_idx)))
        for start in range(0, len(tr_idx), p['batch_size']):
            sel = tr_idx[order[start:start + p['batch_size']].numpy()]
            opt.zero_grad()
            out = net(xb[sel].to(model['device']))
            loss = loss_fn(out, yb[sel].to(model['device']))
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(xb[val_idx].to(model['device'])), yb[val_idx].to(model['device'])).item()
        if v < best - 1e-05:
            best, waited = (v, 0)
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            waited += 1
            if waited >= p['patience']:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

def predict_gru(model, x):
    import torch
    net = model['net']
    net.eval()
    out = []
    xb = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(xb), 4096):
            batch = xb[start:start + 4096].to(model['device'])
            out.append(torch.sigmoid(net(batch)).cpu().numpy())
    return np.concatenate(out)

def TransferResult(label, model, seed, train_epoch, test_epoch, training_size, n_train, n_train_used, n_train_positives, n_oracle_train, n_oracle_positives, n_test, match_note, prevalence_train, prevalence_test, auroc_transferred, auroc_oracle, gap, gap_relative, gap_logit, preprocessor_transferred, preprocessor_oracle=None, predictions=None):
    return {'label': label, 'model': model, 'seed': seed, 'train_epoch': train_epoch, 'test_epoch': test_epoch, 'training_size': training_size, 'n_train': n_train, 'n_train_used': n_train_used, 'n_train_positives': n_train_positives, 'n_oracle_train': n_oracle_train, 'n_oracle_positives': n_oracle_positives, 'n_test': n_test, 'match_note': match_note, 'prevalence_train': prevalence_train, 'prevalence_test': prevalence_test, 'auroc_transferred': auroc_transferred, 'auroc_oracle': auroc_oracle, 'gap': gap, 'gap_relative': gap_relative, 'gap_logit': gap_logit, 'preprocessor_transferred': preprocessor_transferred, 'preprocessor_oracle': [] if preprocessor_oracle is None else preprocessor_oracle, 'predictions': predictions}

def transfer_summary(r):
    return {k: v for k, v in r.items() if k != 'predictions'}

def _logit(p: float) -> float:
    p = min(max(p, 1e-06), 1 - 1e-06)
    return float(np.log(p / (1 - p)))

def run_transfer(fm: FeatureMatrix, y: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray, folds: np.ndarray, make_model, label: str, model_name: str, seed: int, train_epoch: str, test_epoch: str, n_folds: int=5, training_size: str='full', keep_predictions: bool=False) -> TransferResult:
    fm_train_all, y_train_all = (fm_subset(fm, train_mask), y[train_mask])
    fm_test, y_test = (fm_subset(fm, test_mask), y[test_mask])
    folds_test = folds[test_mask]
    n_oracle_train = int(round((n_folds - 1) / n_folds * len(y_test)))
    n_oracle_pos = int(round((n_folds - 1) / n_folds * int(y_test.sum())))
    match_note = None
    if training_size == 'matched' and n_oracle_train < len(y_train_all):
        rng = np.random.default_rng(1000 + seed)
        take = rng.choice(len(y_train_all), size=n_oracle_train, replace=False)
        keep = np.zeros(len(y_train_all), dtype=bool)
        keep[take] = True
        fm_train, y_train = (fm_subset(fm_train_all, keep), y_train_all[keep])
    elif training_size == 'event_matched':
        rng = np.random.default_rng(2000 + seed)
        pos_idx = np.flatnonzero(y_train_all == 1)
        neg_idx = np.flatnonzero(y_train_all == 0)
        want_pos = min(n_oracle_pos, pos_idx.size)
        want_neg = min(n_oracle_train - n_oracle_pos, neg_idx.size)
        if want_pos < n_oracle_pos or want_neg < n_oracle_train - n_oracle_pos:
            match_note = f'training era short of events: wanted {n_oracle_pos} pos / {n_oracle_train - n_oracle_pos} neg, took {want_pos}/{want_neg}'
        keep = np.zeros(len(y_train_all), dtype=bool)
        keep[rng.choice(pos_idx, size=want_pos, replace=False)] = True
        keep[rng.choice(neg_idx, size=want_neg, replace=False)] = True
        fm_train, y_train = (fm_subset(fm_train_all, keep), y_train_all[keep])
    elif training_size in ('full', 'matched'):
        fm_train, y_train = (fm_train_all, y_train_all)
    else:
        raise ValueError(f"training_size must be 'full', 'matched' or 'event_matched', got {training_size!r}")
    pre_train = preprocessor_fit(fm_train, fitted_on=f'epoch:{train_epoch}|size:{training_size}')
    transferred = make_model(seed)
    model_fit_matrix(transferred, fm_train, y_train, pre_train)
    p_transferred = model_predict_matrix(transferred, fm_test, pre_train)
    p_oracle = np.full(len(y_test), np.nan)
    oracle_pre: list[dict] = []
    for k in range(n_folds):
        held = folds_test == k
        if not held.any() or len(np.unique(y_test[~held])) < 2:
            continue
        fm_in, y_in = (fm_subset(fm_test, ~held), y_test[~held])
        pre_fold = preprocessor_fit(fm_in, fitted_on=f'epoch:{test_epoch}|fold:{k}')
        oracle = make_model(seed)
        model_fit_matrix(oracle, fm_in, y_in, pre_fold)
        p_oracle[held] = model_predict_matrix(oracle, fm_subset(fm_test, held), pre_fold)
        oracle_pre.append(preprocessor_describe(pre_fold))
    scored = np.isfinite(p_oracle)
    a_t = auroc(y_test, p_transferred)
    a_o = auroc(y_test[scored], p_oracle[scored])
    gap = a_o - a_t
    return TransferResult(label=label, model=model_name, seed=seed, train_epoch=train_epoch, test_epoch=test_epoch, training_size=training_size, n_train=int(train_mask.sum()), n_train_used=int(len(y_train)), n_train_positives=int(y_train.sum()), n_oracle_train=n_oracle_train, n_oracle_positives=n_oracle_pos, n_test=int(test_mask.sum()), match_note=match_note, prevalence_train=round(float(y_train.mean()), 4), prevalence_test=round(float(y_test.mean()), 4), auroc_transferred=round(float(a_t), 4), auroc_oracle=round(float(a_o), 4), gap=round(float(gap), 4), gap_relative=round(float(gap / max(a_o - 0.5, 1e-06)), 4), gap_logit=round(_logit(a_o) - _logit(a_t), 4), preprocessor_transferred=preprocessor_describe(pre_train), preprocessor_oracle=oracle_pre, predictions={'y': y_test, 'transferred': p_transferred, 'oracle': p_oracle} if keep_predictions else None)







MAX_HOURS = 96.0

def feature_events(con, fcfg: dict, weights: pl.DataFrame) -> dict[str, pl.DataFrame]:
    w = int(fcfg['window_hours'])
    ev: dict[str, pl.DataFrame] = {}
    for name in fcfg['vitals']:
        if name == 'temperature':
            c = chart(con, 'temperature_c', w)
            f = chart(con, 'temperature_f', w).with_columns(value=(pl.col('value') - 32.0) * 5.0 / 9.0)
            ev['temperature'] = pl.concat([c, f]).filter(pl.col('value').is_between(25, 45))
        elif name == 'gcs_total':
            ev['gcs_total'] = gcs_total(con, w)
        else:
            ev[name] = chart(con, name, w)
    for name in fcfg['labs']:
        ev[name] = lab(con, name, w, lookback_days=0).filter(pl.col('hours') >= 0)
    if 'urine_rate' in fcfg['derived']:
        uo = urine_output(con, w).join(weights.select('stay_id', 'weight_kg'), on='stay_id', how='left')
        ev['urine_rate'] = uo.select('stay_id', 'hours', value=pl.col('value') / pl.col('weight_kg'))
    return {k: v.filter(pl.col('hours') < w) for k, v in ev.items()}

def build_main() -> None:
    t0 = time.time()
    lcfg = load_config('labels/primary.yaml')
    fcfg = load_config('features/physiology_only.yaml')
    task, sofa_cfg = (lcfg['task'], lcfg['sofa'])
    T, H = (task['prediction_hour'], task['horizon_hours'])
    window = int(fcfg['window_hours'])
    if window != T:
        raise ValueError(f'feature window {window} must equal prediction hour {T}')
    con = connect()
    build_cohort(con)
    register_source(con, 'mimic_iv', ['labevents', 'chartevents', 'outputevents', 'inputevents', 'prescriptions', 'microbiologyevents'])
    cohort = con.execute('SELECT stay_id, epoch FROM cohort').pl()
    weights_df = weights(con)
    con.register('weights_tmp', weights_df.to_arrow())
    creatinine = lab(con, 'creatinine', MAX_HOURS)
    urine = urine_output(con, MAX_HOURS)
    stays = con.execute(f'SELECT stay_id, least(los_hours, {MAX_HOURS}) AS max_hours FROM cohort').pl()
    sofa = hourly_sofa(SofaInputs(pao2=lab(con, 'pao2', MAX_HOURS), fio2=chart(con, 'fio2', MAX_HOURS), ventilated=ventilated_hours(con, MAX_HOURS), platelets=lab(con, 'platelets', MAX_HOURS), bilirubin=lab(con, 'bilirubin_total', MAX_HOURS), mean_arterial_pressure=chart(con, 'map', MAX_HOURS), gcs_total=gcs_total(con, MAX_HOURS), creatinine=creatinine.filter(pl.col('hours') >= 0), urine=urine, vasopressors=vasopressor_rates(con, MAX_HOURS)), stays)
    suspicion = suspicion_of_infection(antibiotic_times(con, max_hours=MAX_HOURS), culture_times(con, max_hours=MAX_HOURS, variant=lcfg['suspicion']['culture_variant']))
    aki_cr = first_onset(creatinine_stage(creatinine))
    aki_uo = first_onset(urine_output_stage(urine, weights_df.select('stay_id', 'weight_kg')))
    onsets = {'aki_creatinine': aki_cr, 'sofa_dysfunction': organ_dysfunction_onset(sofa, prediction_hour=T, increase=sofa_cfg['increase'], baseline_rule=sofa_cfg['baseline_rule']), 'aki_urine': aki_uo, 'sepsis3': sepsis3_onset(sofa, suspicion, prediction_hour=T, increase=sofa_cfg['increase'], baseline_rule=sofa_cfg['baseline_rule'])}
    prevalent = {'aki_creatinine': prevalent_from_onsets(aki_cr, T), 'aki_urine': prevalent_from_onsets(aki_uo, T), 'sofa_dysfunction': prevalent_dysfunction(sofa, T, sofa_cfg['increase'], sofa_cfg['prevalence_rule']), 'sepsis3': prevalent_sepsis3(sofa, suspicion, T, sofa_cfg['increase'], rule=sofa_cfg['prevalence_rule'])}
    common, common_summary = common_cohort(cohort, prevalent)
    print(f'common cohort: {common.height:,} stays  ({time.time() - t0:.0f}s)')
    ev = feature_events(con, fcfg, weights_df)
    ev = {k: v.filter(pl.col('stay_id').is_in(common['stay_id'])) for k, v in ev.items()}
    assert_no_future_leakage(ev, window)
    variables = list(fcfg['vitals']) + list(fcfg['labs']) + list(fcfg['derived'])
    panel = hourly_panel(ev, common['stay_id'], window)
    static = con.execute("\n        SELECT stay_id, age, gender AS sex,\n               coalesce(admission_type, 'UNKNOWN') AS admission_type\n        FROM cohort\n        ").pl()
    fm = build(panel, static, variables, list(fcfg['aggregations']), window)
    print(f"features: {fm['tabular'].shape} tabular, {fm['sequence'].shape} sequence ({time.time() - t0:.0f}s)")
    order = pl.DataFrame({'stay_id': fm['stay_ids']})
    y = {}
    for name, o in onsets.items():
        joined = order.join(o.select('stay_id', 'onset_hours'), on='stay_id', how='left')
        y[name] = ((joined['onset_hours'] > T) & (joined['onset_hours'] <= H)).fill_null(False).to_numpy().astype(np.int8)
    epochs = order.join(cohort, on='stay_id', how='left')['epoch'].to_numpy().astype('U16')
    folds = assign_folds(fm['stay_ids'], n_folds=5, seed=0)
    out = PROJECT_ROOT / 'data' / 'processed' / 'dataset.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, stay_ids=fm['stay_ids'], tabular=fm['tabular'], sequence=fm['sequence'], static=fm['static'], epochs=epochs, folds=folds, tabular_columns=np.array(fm['tabular_columns']), sequence_columns=np.array(fm['sequence_columns']), static_columns=np.array(fm['static_columns']), **{f'y_{k}': v for k, v in y.items()})
    manifest = {'n_stays': int(fm_n_stays(fm)), 'window_hours': window, 'prediction_hour': T, 'horizon_hours': H, 'n_tabular_features': int(fm['tabular'].shape[1]), 'n_sequence_vars': int(fm['sequence'].shape[2]), 'sequence_shape': list(fm['sequence'].shape[1:]), 'common_cohort': common_summary, 'label_prevalence_by_epoch': [{'epoch': e, **{k: round(float(v[epochs == e].mean()), 4) for k, v in y.items()}, 'n': int((epochs == e).sum())} for e in sorted(set(epochs.tolist()))], 'missing_fraction_tabular': round(float(np.mean(~np.isfinite(fm['tabular']))), 4), 'missing_fraction_sequence': round(float(np.mean(~np.isfinite(fm['sequence']))), 4), 'fold_sizes': {str(k): int((folds == k).sum()) for k in range(5)}}
    path = PROJECT_ROOT / 'results' / 'recon' / 'dataset_manifest.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('\nlabel prevalence by epoch (common cohort):')
    for r in manifest['label_prevalence_by_epoch']:
        print(f"  {r['epoch']:12s} n={r['n']:>6,}  " + '  '.join((f'{k}={r[k]:.3f}' for k in y)))
    print(f"\nmissing: tabular {manifest['missing_fraction_tabular']:.3f}, sequence {manifest['missing_fraction_sequence']:.3f}")
    print(f'-> {out}\n-> {path}   [{time.time() - t0:.0f}s]')

PLAUSIBLE_LOW, PLAUSIBLE_HIGH = (0.6, 0.9)

GRU_TOLERANCE = 0.05

def smoke_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--models', default='logreg,xgboost,gru')
    ap.add_argument('--training-sizes', default='full,matched')
    args = ap.parse_args()
    t0 = time.time()
    assert_target_epochs_sealed((SMOKE_TRAIN_EPOCH, SMOKE_TEST_EPOCH))
    mcfg = load_config('models/smoke.yaml')
    ds = dataset_load(PROJECT_ROOT / 'data' / 'processed' / 'dataset.npz')
    fm = dataset_matrix(ds)
    train_mask = dataset_epoch_mask(ds, SMOKE_TRAIN_EPOCH)
    test_mask = dataset_epoch_mask(ds, SMOKE_TEST_EPOCH)
    assert_disjoint(ds['stay_ids'][train_mask], ds['stay_ids'][test_mask])
    assert_folds_partition(ds['stay_ids'][test_mask], ds['folds'][test_mask], 5)
    print(f'train {SMOKE_TRAIN_EPOCH}: {train_mask.sum():,}   test {SMOKE_TEST_EPOCH}: {test_mask.sum():,}')
    rows = []
    for size in args.training_sizes.split(','):
        for model_name in args.models.split(','):
            params = mcfg[model_name]
            for label in LABELS:
                for seed in range(args.seeds):
                    r = run_transfer(fm=fm, y=dataset_y(ds, label), train_mask=train_mask, test_mask=test_mask, folds=ds['folds'], make_model=lambda s, m=model_name, p=params: build_model(m, seed=s, **p), label=label, model_name=model_name, seed=seed, train_epoch=SMOKE_TRAIN_EPOCH, test_epoch=SMOKE_TEST_EPOCH, training_size=size)
                    rows.append(transfer_summary(r))
                    print(f"  {size:7s} {model_name:8s} {label:17s} seed={seed}  n_tr={r['n_train_used']:>6,}  transferred={r['auroc_transferred']:.3f}  oracle={r['auroc_oracle']:.3f}  gap={r['gap']:+.4f}  [{time.time() - t0:.0f}s]")
    checks = sanity_checks(rows)
    out = {'train_epoch': SMOKE_TRAIN_EPOCH, 'test_epoch': SMOKE_TEST_EPOCH, 'seeds': args.seeds, 'model_params': {m: mcfg[m] for m in args.models.split(',')}, 'results': rows, 'sanity': checks, 'elapsed_seconds': round(time.time() - t0, 1)}
    path = PROJECT_ROOT / 'results' / 'smoke' / 'smoke_run.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float), encoding='utf-8')
    report(rows, checks)
    print(f'\n-> {path}   [{time.time() - t0:.0f}s]')
    if any((c['status'] == 'FAIL' for c in checks)):
        raise SystemExit(1)

def sanity_checks(rows: list[dict]) -> list[dict]:
    out = []

    def by(model=None, label=None, size=None):
        return [r for r in rows if (model is None or r['model'] == model) and (label is None or r['label'] == label) and (size is None or r['training_size'] == size)]
    sizes = {r['training_size'] for r in rows}
    if {'full', 'matched'} <= sizes:
        detail = []
        for model in {r['model'] for r in rows}:
            full = float(np.mean([r['gap'] for r in by(model, size='full')]))
            matched = float(np.mean([r['gap'] for r in by(model, size='matched')]))
            detail.append({'model': model, 'mean_gap_full': round(full, 4), 'mean_gap_matched': round(matched, 4), 'volume_component': round(matched - full, 4)})
        bad = [d for d in detail if d['volume_component'] < -0.001]
        out.append({'check': 'size_matching_does_not_lower_the_gap', 'value': detail, 'status': 'OK' if not bad else 'REVIEW', 'note': 'volume_component = matched - full is the part of the full gap that was training volume rather than era; it should be non-negative and larger for more flexible models'})
    for label in {r['label'] for r in rows}:
        aurocs = [r['auroc_oracle'] for r in by(label=label)]
        lo, hi = (min(aurocs), max(aurocs))
        out.append({'check': f'auroc_plausible[{label}]', 'value': [round(lo, 3), round(hi, 3)], 'status': 'OK' if PLAUSIBLE_LOW <= lo and hi <= PLAUSIBLE_HIGH else 'REVIEW', 'note': f'oracle AUROC across models/seeds; band [{PLAUSIBLE_LOW}, {PLAUSIBLE_HIGH}]'})
    leaks = [r for r in rows if max(r['auroc_oracle'], r['auroc_transferred']) > 0.9]
    out.append({'check': 'no_auroc_above_0.90', 'value': [{'label': r['label'], 'model': r['model'], 'auroc': max(r['auroc_oracle'], r['auroc_transferred'])} for r in leaks], 'status': 'OK' if not leaks else 'FAIL', 'note': 'AUROC above 0.90 is treated as leakage until explained'})
    models = {r['model'] for r in rows}
    if {'gru', 'xgboost'} <= models:
        worst = []
        for label in {r['label'] for r in rows}:
            g = np.mean([r['auroc_oracle'] for r in by('gru', label)])
            x = np.mean([r['auroc_oracle'] for r in by('xgboost', label)])
            worst.append({'label': label, 'gru': round(float(g), 3), 'xgboost': round(float(x), 3), 'deficit': round(float(x - g), 3)})
        bad = [w for w in worst if w['deficit'] > GRU_TOLERANCE]
        out.append({'check': 'gru_not_grossly_behind_xgboost', 'value': worst, 'status': 'OK' if not bad else 'FAIL', 'note': f'a deficit above {GRU_TOLERANCE} AUROC suggests a malformed sequence tensor rather than a weak architecture'})
    seed_sd = []
    for model in models:
        for label in {r['label'] for r in rows}:
            gaps = [r['gap'] for r in by(model, label)]
            if len(gaps) > 1:
                seed_sd.append({'model': model, 'label': label, 'sd': round(float(np.std(gaps, ddof=1)), 4)})
    if seed_sd:
        worst_sd = max((s['sd'] for s in seed_sd))
        planned_seeds = 5
        se_at_plan = round(worst_sd / np.sqrt(planned_seeds), 4)
        out.append({'check': 'seed_noise_below_half_sesoi', 'value': {'max_sd': worst_sd, 'planned_seeds': planned_seeds, 'se_of_mean_at_plan': se_at_plan, 'detail': seed_sd}, 'status': 'OK' if se_at_plan < 0.005 else 'REVIEW', 'note': 'judged on the standard error of the seed-averaged Gap, sd/sqrt(seeds), which is what adding seeds actually reduces. Estimated from 2 seeds here (1 degree of freedom), so it is indicative only; PREREGISTRATION §9 amended accordingly'})
    return out

def report(rows: list[dict], checks: list[dict]) -> None:
    print('\nmean over seeds:')
    print(f"{'size':8s} {'model':9s} {'label':18s} {'transf':>7s} {'oracle':>7s} {'gap':>8s} {'prev':>6s}")
    for size in dict.fromkeys((r['training_size'] for r in rows)):
        for model in dict.fromkeys((r['model'] for r in rows)):
            for label in LABELS:
                sel = [r for r in rows if r['model'] == model and r['label'] == label and (r['training_size'] == size)]
                if not sel:
                    continue
                print(f"{size:8s} {model:9s} {label:18s} {np.mean([r['auroc_transferred'] for r in sel]):>7.3f} {np.mean([r['auroc_oracle'] for r in sel]):>7.3f} {np.mean([r['gap'] for r in sel]):>+8.4f} {np.mean([r['prevalence_test'] for r in sel]):>6.3f}")
    print('\nsanity:')
    for c in checks:
        print(f"  [{c['status']:6s}] {c['check']}")
        if c['status'] != 'OK':
            print(f"           {c['value']}")

STAGES = {
    'build': build_main,
    'smoke': smoke_main,
}

if __name__ == '__main__':
    argv = sys.argv[1:]
    chosen = [argv[0]] if argv and argv[0] in STAGES else list(STAGES)
    rest = argv[1:] if argv and argv[0] in STAGES else argv
    for name in chosen:
        sys.argv = [name, *rest]
        print('== ' + name + ' ==')
        STAGES[name]()
