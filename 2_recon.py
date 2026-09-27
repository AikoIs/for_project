from __future__ import annotations
import copy
import sys
import yaml
from datetime import datetime
from pathlib import Path
from scipy.stats import norm
import argparse
import duckdb
import json
import numpy as np
import polars as pl
import time
import yaml

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


def auroc_batch(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(bool)
    s = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    k, n = s.shape
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(k, np.nan)
    order = np.argsort(s, axis=1, kind='stable')
    xs = np.take_along_axis(s, order, axis=1)
    is_new = np.ones((k, n), dtype=bool)
    np.not_equal(xs[:, 1:], xs[:, :-1], out=is_new[:, 1:])
    is_last = np.ones((k, n), dtype=bool)
    is_last[:, :-1] = is_new[:, 1:]
    idx = np.arange(1, n + 1, dtype=np.float64)[None, :]
    start = np.maximum.accumulate(np.where(is_new, idx, 0.0), axis=1)
    end = np.minimum.accumulate(np.where(is_last, idx, n + 1.0)[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty((k, n), dtype=np.float64)
    np.put_along_axis(ranks, order, (start + end) / 2.0, axis=1)
    pos_rank_sum = ranks[:, y].sum(axis=1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

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

def creatinine_stage(creatinine, baseline=None) -> pl.DataFrame:
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
        raise ValueError('unknown baseline method ' + repr(baseline['method']))
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

PREDICTION_HOUR = 6.0

HORIZONS = (24.0, 48.0, 72.0)

def summarise_onsets(onsets: pl.DataFrame, cohort: pl.DataFrame, label: str, prediction_hour: float=PREDICTION_HOUR, horizons: tuple[float, ...]=HORIZONS) -> list[dict]:
    df = cohort.join(onsets.select('stay_id', 'onset_hours'), on='stay_id', how='left')
    rows = []
    for _key, g in sorted(df.group_by('epoch'), key=lambda kv: kv[0]):
        epoch = g['epoch'][0]
        n = g.height
        onset = g['onset_hours']
        ever = onset.is_not_null()
        early = ever & (onset <= prediction_hour)
        n_ever, n_early = (int(ever.sum()), int(early.sum()))
        eligible = n - n_early
        rec = {'label': label, 'epoch': epoch, 'n_stays': n, 'n_ever_onset': n_ever, 'n_onset_before_6h': n_early, 'frac_onset_before_6h_of_all_onsets': round(n_early / n_ever, 4) if n_ever else None, 'n_eligible_after_excluding_early': eligible, 'frac_cohort_lost_to_early_onset': round(n_early / n, 4)}
        for h in horizons:
            inside = int((ever & (onset > prediction_hour) & (onset <= h)).sum())
            rec[f'n_events_6_to_{int(h)}h'] = inside
            rec[f'prevalence_6_to_{int(h)}h'] = round(inside / eligible, 4) if eligible else None
        q = onset.filter(ever)
        for name, p in (('q25', 0.25), ('median', 0.5), ('q75', 0.75)):
            rec[f'onset_hours_{name}'] = round(float(q.quantile(p)), 2) if q.len() else None
        rows.append(rec)
    return rows

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

def hourly_sofa(inputs, stays) -> pl.DataFrame:
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

C_GRID = np.linspace(0.02, 0.995, 60)

def build_c_lookup(prevalence: float, rng: np.random.Generator, n_mc: int=400000):
    t = norm.isf(prevalence)
    u = rng.standard_normal(n_mc)
    y = (u > t).astype(np.int8)
    w = rng.standard_normal(n_mc)
    aucs = np.array([auroc(y, c * u + np.sqrt(1 - c * c) * w) for c in C_GRID])
    return (aucs, C_GRID)

def c_for(lookup, target_auc: float) -> float:
    aucs, cs = lookup
    return float(np.interp(np.clip(target_auc, aucs[0], aucs[-1]), aucs, cs))

def _scores_for_label(u, y, lookup, oracle_auc, gap, cfg, rng):
    n = u.size
    n_models, n_seeds, n_folds = (cfg['n_models'], cfg['n_seeds'], cfg['n_folds'])
    fold = rng.permutation(n) % n_folds
    orc_rows, trn_rows = ([], [])
    for m in range(n_models):
        m_jit = rng.normal(0, cfg['model_skill_sd'])
        for _s in range(n_seeds):
            s_jit = rng.normal(0, cfg['seed_skill_sd'])
            base = oracle_auc + m_jit + s_jit
            w = rng.standard_normal(n)
            orc = np.empty(n)
            for k in range(n_folds):
                c = c_for(lookup, base + rng.normal(0, cfg['fold_skill_sd']))
                sel = fold == k
                orc[sel] = c * u[sel] + np.sqrt(1 - c * c) * w[sel]
            orc_rows.append(orc)
            c_t = c_for(lookup, base - gap)
            w2 = rng.standard_normal(n)
            trn_rows.append(c_t * u + np.sqrt(1 - c_t * c_t) * w2)
    return (np.asarray(orc_rows), np.asarray(trn_rows), y)

def bootstrap_did_se(contrast, n, gap_b, true_did, cfg, lookups, rng, eval_idx=None):
    r = contrast['latent_corr']
    g = rng.standard_normal(n)
    payload = {}
    for role, gap in (('a', gap_b + true_did), ('b', gap_b)):
        spec = contrast[f'label_{role}']
        u = np.sqrt(r) * g + np.sqrt(1 - r) * rng.standard_normal(n)
        y = (u > norm.isf(spec['prevalence'])).astype(np.int8)
        payload[role] = _scores_for_label(u, y, lookups[spec['name']], spec['oracle_auroc'], gap, cfg, rng)
    if eval_idx is None:
        eval_idx = np.arange(n)
    n_eval = eval_idx.size
    boot = np.empty(cfg['n_bootstrap'])
    for b in range(cfg['n_bootstrap']):
        take = eval_idx[rng.integers(0, n_eval, n_eval)]
        gaps = {}
        for role in ('a', 'b'):
            orc, trn, y = payload[role]
            yb = y[take]
            a_orc = auroc_batch(yb, orc[:, take])
            a_trn = auroc_batch(yb, trn[:, take])
            gaps[role] = np.nanmean(a_orc - a_trn)
        boot[b] = gaps['a'] - gaps['b']
    return boot

def analytic_power(se: float, true_did: float, sesoi: float, alpha: float) -> dict:
    z = norm.isf(alpha / 2)
    thresh = max(sesoi, z * se)
    p_meaningful = float(norm.sf((thresh - true_did) / se))
    room = sesoi - norm.isf(0.05) * se
    p_equivalent = 0.0 if room <= 0 else float(norm.cdf((room - true_did) / se) - norm.cdf((-room - true_did) / se))
    return {'p_meaningful': round(p_meaningful, 3), 'p_equivalent': round(p_equivalent, 3), 'detection_threshold': round(thresh, 4)}

def power_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='tiny smoke run to check the code path, not a real estimate')
    ap.add_argument('--config', default='experiments/power.yaml')
    ap.add_argument('--out', default='power_analysis.json')
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg |= {'n_bootstrap': 30, 'n_replicates': 1, 'epoch_sizes': [4000]}
    rng = np.random.default_rng(cfg['seed'])
    sesoi = cfg['sesoi']
    lookups = {}
    for contrast in cfg['contrasts']:
        for role in ('a', 'b'):
            spec = contrast[f'label_{role}']
            if spec['name'] not in lookups:
                lookups[spec['name']] = build_c_lookup(spec['prevalence'], rng)
    results = []
    t_start = time.time()
    for contrast in cfg['contrasts']:
        for n in cfg['epoch_sizes']:
            for design in ('crossfit', 'holdout'):
                ses = []
                for _ in range(cfg['n_replicates']):
                    idx = None
                    if design == 'holdout':
                        keep = int(round(n * cfg['holdout_fraction']))
                        idx = rng.choice(n, size=keep, replace=False)
                    boot = bootstrap_did_se(contrast, n, cfg['gap_b'], sesoi, cfg, lookups, rng, idx)
                    ses.append(float(np.nanstd(boot, ddof=1)))
                se = float(np.mean(ses))
                row = {'contrast': contrast['name'], 'design': design, 'n_epoch': n, 'n_eval': n if design == 'crossfit' else int(round(n * cfg['holdout_fraction'])), 'se_did': round(se, 5), 'ci95_halfwidth': round(1.96 * se, 5), 'expected_delta_min_at_null': round(2.443 * se, 5), 'power': {f'did={d}': {'alpha_0.05': analytic_power(se, d, sesoi, 0.05), 'alpha_holm_worst': analytic_power(se, d, sesoi, 0.05 / 12)} for d in cfg['true_did']}}
                results.append(row)
                print(f"{contrast['name']:12s} {design:9s} n={n:6d} se={se:.4f} CI95=+-{1.96 * se:.4f} delta_min~{2.443 * se:.4f}")
    out = {'config': cfg, 'sesoi': sesoi, 'elapsed_seconds': round(time.time() - t_start, 1), 'results': results, 'note': 'Placeholder prevalences/epoch sizes until phase-0 reconnaissance supplies real ones; rerun with placeholder=false before freezing the preregistration.'}
    path = PROJECT_ROOT / 'results' / 'power' / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\n-> {path}')

aki_PREDICTION_HOUR = 6.0

aki_HORIZONS = [24.0, 48.0, 72.0]


def _fetch_creatinine(con, itemids: list[int]) -> pl.DataFrame:
    ids = ', '.join(map(str, itemids))
    return con.execute(f"\n        SELECT c.stay_id,\n               date_diff('second', c.intime, l.charttime) / 3600.0 AS hours,\n               l.valuenum AS value\n        FROM mimic_iv_labevents l\n        JOIN cohort c ON c.hadm_id = l.hadm_id\n        WHERE l.itemid IN ({ids})\n          AND l.valuenum IS NOT NULL\n          AND l.valuenum BETWEEN 0.1 AND 50\n          AND l.charttime BETWEEN c.intime - INTERVAL 7 DAY\n                              AND c.intime + INTERVAL 96 HOUR\n        ").pl()

def _fetch_urine(con, add_ids: list[int], sub_ids: list[int]) -> pl.DataFrame:
    add = ', '.join(map(str, add_ids))
    sub = ', '.join(map(str, sub_ids)) or '-1'
    return con.execute(f"\n        SELECT o.stay_id,\n               date_diff('second', c.intime, o.charttime) / 3600.0 AS hours,\n               sum(CASE WHEN o.itemid IN ({sub}) THEN -o.value ELSE o.value END) AS value\n        FROM mimic_iv_outputevents o\n        JOIN cohort c ON c.stay_id = o.stay_id\n        WHERE o.itemid IN ({add}, {sub})\n          AND o.value IS NOT NULL AND o.value BETWEEN 0 AND 5000\n          AND o.charttime >= c.intime\n          AND o.charttime <= c.intime + INTERVAL 96 HOUR\n        GROUP BY 1, 2\n        ").pl()

def _fetch_weights(con) -> pl.DataFrame:
    raw = con.execute('\n        SELECT c.stay_id, c.epoch, c.gender,\n               CASE WHEN e.itemid = 226531 THEN e.valuenum * 0.45359237\n                    ELSE e.valuenum END AS w,\n               e.charttime\n        FROM mimic_iv_chartevents e\n        JOIN cohort c ON c.stay_id = e.stay_id\n        WHERE e.itemid IN (226512, 224639, 226531)\n          AND e.valuenum IS NOT NULL\n          AND e.charttime <= c.intime + INTERVAL 48 HOUR\n        ').pl()
    first = raw.filter(pl.col('w').is_between(20, 400)).sort(['stay_id', 'charttime']).group_by('stay_id').agg(weight_kg=pl.col('w').first())
    cohort = con.execute('SELECT stay_id, epoch, gender FROM cohort').pl()
    joined = cohort.join(first, on='stay_id', how='left')
    med = joined.group_by('gender').agg(fallback=pl.col('weight_kg').median())
    joined = joined.join(med, on='gender', how='left').with_columns(weight_imputed=pl.col('weight_kg').is_null(), weight_kg=pl.col('weight_kg').fill_null(pl.col('fallback')))
    return joined.select('stay_id', 'epoch', 'weight_kg', 'weight_imputed')

def summarise(onsets: pl.DataFrame, cohort: pl.DataFrame, label: str) -> list[dict]:
    df = cohort.join(onsets, on='stay_id', how='left')
    rows = []
    for _key, g in sorted(df.group_by('epoch'), key=lambda kv: kv[0]):
        epoch = g['epoch'][0]
        n = g.height
        onset = g['onset_hours']
        ever = onset.is_not_null()
        early = ever & (onset <= aki_PREDICTION_HOUR)
        rec = {'label': label, 'epoch': epoch, 'n_stays': n, 'n_ever_onset_96h': int(ever.sum()), 'n_onset_before_6h': int(early.sum()), 'frac_onset_before_6h_of_all_onsets': round(float(early.sum()) / float(ever.sum()), 4) if ever.sum() else None, 'n_eligible_after_excluding_early': int(n - early.sum()), 'frac_cohort_lost_to_early_onset': round(float(early.sum()) / n, 4)}
        for h in aki_HORIZONS:
            inside = ever & (onset > aki_PREDICTION_HOUR) & (onset <= h)
            eligible = n - int(early.sum())
            rec[f'n_events_6_to_{int(h)}h'] = int(inside.sum())
            rec[f'prevalence_6_to_{int(h)}h'] = round(float(inside.sum()) / eligible, 4) if eligible else None
        q = onset.filter(ever)
        rec['onset_hours_q25'] = round(float(q.quantile(0.25)), 2) if q.len() else None
        rec['onset_hours_median'] = round(float(q.median()), 2) if q.len() else None
        rec['onset_hours_q75'] = round(float(q.quantile(0.75)), 2) if q.len() else None
        rows.append(rec)
    return rows

def aki_main() -> None:
    global MIN_CELL
    MIN_CELL = int(paths()['min_cell_count'])
    cfg = load_config('concepts/mimic_iv.yaml')
    con = connect()
    build_cohort(con)
    register_source(con, 'mimic_iv', ['labevents', 'chartevents', 'outputevents'])
    cohort = con.execute('SELECT stay_id, epoch FROM cohort').pl()
    print(f'cohort: {cohort.height:,} stays')
    weights = _fetch_weights(con)
    print(f"weight imputed for {weights['weight_imputed'].mean():.1%} of stays")
    cr = _fetch_creatinine(con, cfg['labs']['creatinine']['itemids'])
    print(f'creatinine rows: {cr.height:,}')
    cr_onset = first_onset(creatinine_stage(cr))
    print(f'  stays with KDIGO-Cr stage >= 1 within 96 h: {cr_onset.height:,}')
    uo_cfg = cfg['output']['urine_output']
    uo = _fetch_urine(con, uo_cfg['itemids'], uo_cfg.get('subtract_itemids', []))
    print(f'urine output rows: {uo.height:,}')
    uo_onset = first_onset(urine_output_stage(uo, weights.select('stay_id', 'weight_kg')))
    print(f'  stays with KDIGO-UO stage >= 1 within 96 h: {uo_onset.height:,}')
    report = {'prediction_hour': aki_PREDICTION_HOUR, 'horizons': aki_HORIZONS, 'weight_imputed_fraction': round(float(weights['weight_imputed'].mean()), 4), 'labels': summarise(cr_onset, cohort, 'aki_creatinine') + summarise(uo_onset, cohort, 'aki_urine')}
    both = cohort.join(cr_onset.rename({'onset_hours': 'cr_h'}), on='stay_id', how='left').join(uo_onset.rename({'onset_hours': 'uo_h'}), on='stay_id', how='left').with_columns(early=(pl.col('cr_h') <= aki_PREDICTION_HOUR).fill_null(False) | (pl.col('uo_h') <= aki_PREDICTION_HOUR).fill_null(False))
    report['common_cohort_aki_only'] = [{'epoch': g['epoch'][0], 'n_stays': g.height, 'n_after_excluding_either_early': int((~g['early']).sum()), 'frac_retained': round(float((~g['early']).mean()), 4)} for _key, g in sorted(both.group_by('epoch'), key=lambda kv: kv[0])]
    out = PROJECT_ROOT / 'results' / 'recon' / 'aki_events.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nlabel              epoch        n     ever  <=6h   (6,48]  prev48  medOnset')
    for r in report['labels']:
        print(f"{r['label']:18s} {r['epoch']:12s} {r['n_stays']:>6,} {r['n_ever_onset_96h']:>6,} {r['n_onset_before_6h']:>6,} {r['n_events_6_to_48h']:>7,} {(r['prevalence_6_to_48h'] if r['prevalence_6_to_48h'] is not None else 0):>6.3f} {(r['onset_hours_median'] if r['onset_hours_median'] is not None else -1):>8.1f}")
    print('\ncommon cohort (AKI labels only):')
    for r in report['common_cohort_aki_only']:
        print(f"  {r['epoch']:12s} {r['n_stays']:>6,} -> {r['n_after_excluding_either_early']:>6,} ({r['frac_retained']:.2f})")
    print(f'\n-> {out}')

SIGNALS = [('proc_invasive_vent', 'procedureevents', 225792), ('proc_noninvasive_vent', 'procedureevents', 225794), ('proc_extubation', 'procedureevents', 227194), ('chart_vent_mode', 'chartevents', 223849), ('chart_vent_type', 'chartevents', 223848), ('chart_o2_device', 'chartevents', 226732), ('chart_peep_set', 'chartevents', 220339), ('chart_tidal_vol_obs', 'chartevents', 224685), ('chart_tidal_vol_set', 'chartevents', 224684), ('chart_minute_vol', 'chartevents', 224687), ('chart_insp_time', 'chartevents', 224738), ('chart_fio2', 'chartevents', 223835), ('chart_resp_rate_set', 'chartevents', 224688)]

WINDOW_HOURS = 48

def ventilation_main() -> None:
    min_cell = int(paths()['min_cell_count'])
    con = connect()
    build_cohort(con)
    register_source(con, 'mimic_iv', ['chartevents', 'procedureevents', 'd_items'])
    labels = dict(con.execute('SELECT itemid, label FROM mimic_iv_d_items').fetchall())
    epochs = [r[0] for r in con.execute('SELECT DISTINCT epoch FROM cohort ORDER BY epoch').fetchall()]
    n_by_epoch = dict(con.execute('SELECT epoch, count(*) FROM cohort GROUP BY 1').fetchall())
    results = []
    for concept, table, itemid in SIGNALS:
        time_col = 'starttime' if table == 'procedureevents' else 'charttime'
        rows = con.execute(f'\n            WITH hit AS (\n                SELECT DISTINCT c.stay_id, c.epoch\n                FROM mimic_iv_{table} e\n                JOIN cohort c ON c.stay_id = e.stay_id\n                WHERE e.itemid = {itemid}\n                  AND e.{time_col} >= c.intime\n                  AND e.{time_col} <= c.intime + INTERVAL {WINDOW_HOURS} HOUR\n            )\n            SELECT c.epoch, count(h.stay_id)\n            FROM cohort c LEFT JOIN hit h USING (stay_id)\n            GROUP BY c.epoch ORDER BY c.epoch\n            ').fetchall()
        by_epoch = {e: int(n) for e, n in rows}
        rec = {'concept': concept, 'table': table, 'itemid': itemid, 'label': labels.get(itemid), 'coverage': {e: {'n_stays': n_by_epoch[e], 'n_with_signal': None if 0 < by_epoch.get(e, 0) < min_cell else by_epoch.get(e, 0), 'frac': round(by_epoch.get(e, 0) / n_by_epoch[e], 4)} for e in epochs}}
        results.append(rec)
    hdr = '  '.join((f'{e[-4:]:>6s}' for e in epochs))
    print(f"{'concept':22s} {'itemid':>7s}  {hdr}   label")
    for r in results:
        line = '  '.join((f"{r['coverage'][e]['frac']:6.3f}" for e in epochs))
        print(f"{r['concept']:22s} {r['itemid']:>7d}  {line}   {r['label']}")
    ref, last = (epochs[2], epochs[-1])
    print(f'\nratio {last} / {ref}:')
    verdict_rows = []
    for r in results:
        a, b = (r['coverage'][ref]['frac'], r['coverage'][last]['frac'])
        ratio = round(b / a, 3) if a > 0 else None
        verdict_rows.append({'concept': r['concept'], 'ratio': ratio})
        print(f"  {r['concept']:22s} {(ratio if ratio is not None else float('nan')):6.3f}")
    out = {'window_hours': WINDOW_HOURS, 'epochs': epochs, 'signals': results, 'last_vs_2014_16_ratio': verdict_rows}
    path = PROJECT_ROOT / 'results' / 'recon' / 'ventilation_check.json'
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\n-> {path}')

labels_MAX_HOURS = 96.0

def labels_main() -> None:
    t0 = time.time()
    con = connect()
    build_cohort(con)
    register_source(con, 'mimic_iv', ['labevents', 'chartevents', 'outputevents', 'inputevents', 'prescriptions', 'microbiologyevents'])
    cohort = con.execute('SELECT stay_id, epoch FROM cohort').pl()
    stays = con.execute(f'SELECT stay_id, least(los_hours, {labels_MAX_HOURS}) AS max_hours FROM cohort').pl()
    print(f'cohort: {cohort.height:,} stays')
    w = weights(con)
    con.register('weights_tmp', w.to_arrow())
    print(f"weight imputed for {w['weight_imputed'].mean():.1%} of stays")

    def step(name, fn):
        t = time.time()
        out = fn()
        print(f'  {name:22s} {out.height:>10,} rows  ({time.time() - t:.0f}s)')
        return out
    print('extracting concepts:')
    creatinine = step('creatinine', lambda: lab(con, 'creatinine', labels_MAX_HOURS))
    platelets = step('platelets', lambda: lab(con, 'platelets', labels_MAX_HOURS))
    bilirubin = step('bilirubin', lambda: lab(con, 'bilirubin_total', labels_MAX_HOURS))
    pao2 = step('pao2', lambda: lab(con, 'pao2', labels_MAX_HOURS))
    fio2 = step('fio2', lambda: chart(con, 'fio2', labels_MAX_HOURS))
    mapp = step('map', lambda: chart(con, 'map', labels_MAX_HOURS))
    gcs = step('gcs_total', lambda: gcs_total(con, labels_MAX_HOURS))
    vent = step('ventilated', lambda: ventilated_hours(con, labels_MAX_HOURS))
    urine = step('urine_output', lambda: urine_output(con, labels_MAX_HOURS))
    pressors = step('vasopressors', lambda: vasopressor_rates(con, labels_MAX_HOURS))
    abx = step('antibiotics', lambda: antibiotic_times(con, max_hours=labels_MAX_HOURS))
    cultures = step('cultures', lambda: culture_times(con, max_hours=labels_MAX_HOURS))
    print('scoring SOFA ...')
    t = time.time()
    sofa = hourly_sofa(SofaInputs(pao2=pao2, fio2=fio2, ventilated=vent, platelets=platelets, bilirubin=bilirubin, mean_arterial_pressure=mapp, gcs_total=gcs, creatinine=creatinine.filter(pl.col('hours') >= 0), urine=urine, vasopressors=pressors), stays)
    print(f'  {sofa.height:,} stay-hours ({time.time() - t:.0f}s)')
    onsets = {'aki_creatinine': first_onset(creatinine_stage(creatinine)), 'sofa_dysfunction': organ_dysfunction_onset(sofa), 'aki_urine': first_onset(urine_output_stage(urine, w.select('stay_id', 'weight_kg'))), 'sepsis3': sepsis3_onset(sofa, suspicion_of_infection(abx, cultures))}
    report = {'max_hours': labels_MAX_HOURS, 'n_cohort': cohort.height, 'weight_imputed_fraction': round(float(w['weight_imputed'].mean()), 4), 'suspicion_pairs': suspicion_of_infection(abx, cultures).height, 'labels': []}
    for label, o in onsets.items():
        report['labels'] += summarise_onsets(o, cohort, label)
    _, common = common_cohort(cohort, onsets)
    report['common_cohort'] = common
    comp = sofa.filter(pl.col('hour') == 6).join(cohort, on='stay_id', how='inner').group_by('epoch').agg([pl.col(c).is_not_null().mean().round(4).alias(c) for c in ('sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal')] + [pl.len().alias('n')]).sort('epoch')
    report['sofa_component_availability_at_hour_6'] = comp.to_dicts()
    out = PROJECT_ROOT / 'results' / 'recon' / 'label_events.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nlabel              epoch            n    ever   <=6h  (6,48]   prev  medOnset')
    for r in report['labels']:
        print(f"{r['label']:18s} {r['epoch']:12s} {r['n_stays']:>6,} {r['n_ever_onset']:>7,} {r['n_onset_before_6h']:>6,} {r['n_events_6_to_48h']:>7,} {r['prevalence_6_to_48h'] or 0:>6.3f} {(r['onset_hours_median'] if r['onset_hours_median'] is not None else -1):>8.1f}")
    print('\nshare of all onsets that occur at or before hour 6:')
    for label in onsets:
        fr = [r['frac_onset_before_6h_of_all_onsets'] for r in report['labels'] if r['label'] == label]
        print(f'  {label:18s} ' + ' '.join((f'{f:.2f}' if f else '  . ' for f in fr)))
    print('\ncommon cohort across all four labels:')
    for r in common:
        print(f"  {r['epoch']:12s} {r['n_stays']:>6,} -> {r['n_common_cohort']:>6,} ({r['frac_retained']:.2f})")
    print('\nSOFA component availability at hour 6:')
    print(comp)
    print(f'\n-> {out}   [{time.time() - t0:.0f}s total]')

design_MAX_HOURS = 96.0

design_BASELINE_RULES = ['at_prediction', 'min_before', 'zero']

PREDICTION_HOURS = [4.0, 6.0]

design_HORIZONS = [48.0, 72.0]

design_TARGET_EPOCHS = ['2014 - 2016', '2017 - 2019', '2020 - 2022']

def design_main() -> None:
    t0 = time.time()
    con = connect()
    build_cohort(con)
    register_source(con, 'mimic_iv', ['labevents', 'chartevents', 'outputevents', 'inputevents', 'prescriptions', 'microbiologyevents'])
    cohort = con.execute('SELECT stay_id, epoch FROM cohort').pl()
    stays = con.execute(f'SELECT stay_id, least(los_hours, {design_MAX_HOURS}) AS max_hours FROM cohort').pl()
    w = weights(con)
    con.register('weights_tmp', w.to_arrow())
    creatinine = lab(con, 'creatinine', design_MAX_HOURS)
    urine = urine_output(con, design_MAX_HOURS)
    sofa = hourly_sofa(SofaInputs(pao2=lab(con, 'pao2', design_MAX_HOURS), fio2=chart(con, 'fio2', design_MAX_HOURS), ventilated=ventilated_hours(con, design_MAX_HOURS), platelets=lab(con, 'platelets', design_MAX_HOURS), bilirubin=lab(con, 'bilirubin_total', design_MAX_HOURS), mean_arterial_pressure=chart(con, 'map', design_MAX_HOURS), gcs_total=gcs_total(con, design_MAX_HOURS), creatinine=creatinine.filter(pl.col('hours') >= 0), urine=urine, vasopressors=vasopressor_rates(con, design_MAX_HOURS)), stays)
    suspicion = suspicion_of_infection(antibiotic_times(con, max_hours=design_MAX_HOURS), culture_times(con, max_hours=design_MAX_HOURS))
    print(f'inputs ready ({time.time() - t0:.0f}s); {suspicion.height:,} suspicion pairs')
    aki_cr = first_onset(creatinine_stage(creatinine))
    aki_uo = first_onset(urine_output_stage(urine, w.select('stay_id', 'weight_kg')))
    cells = []
    for rule in design_BASELINE_RULES:
        for ph in PREDICTION_HOURS:
            onsets = {'aki_creatinine': aki_cr, 'sofa_dysfunction': organ_dysfunction_onset(sofa, prediction_hour=ph, baseline_rule=rule), 'aki_urine': aki_uo, 'sepsis3': sepsis3_onset(sofa, suspicion, prediction_hour=ph, baseline_rule=rule)}
            prevalent = {'aki_creatinine': prevalent_from_onsets(aki_cr, ph), 'aki_urine': prevalent_from_onsets(aki_uo, ph)}
            _, common = common_cohort(cohort, prevalent)
            common_by_epoch = {r['epoch']: r for r in common}
            for horizon in design_HORIZONS:
                cell = {'baseline_rule': rule, 'prediction_hour': ph, 'horizon': horizon, 'epochs': {}}
                for label, o in onsets.items():
                    rows = summarise_onsets(o, cohort, label, prediction_hour=ph, horizons=(horizon,))
                    for r in rows:
                        e = cell['epochs'].setdefault(r['epoch'], {'n_common_cohort': common_by_epoch[r['epoch']]['n_common_cohort'], 'frac_retained': common_by_epoch[r['epoch']]['frac_retained'], 'labels': {}})
                        e['labels'][label] = {'n_events': r[f'n_events_6_to_{int(horizon)}h'], 'prevalence': r[f'prevalence_6_to_{int(horizon)}h'], 'frac_onsets_before_prediction': r['frac_onset_before_6h_of_all_onsets'], 'onset_median': r['onset_hours_median']}
                cells.append(cell)
    out = {'max_hours': design_MAX_HOURS, 'cells': cells}
    path = PROJECT_ROOT / 'results' / 'recon' / 'design_options.json'
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('\ncommon cohort in the three target epochs, and events in the window')
    print(f"{'baseline':14s} {'T':>4s} {'H':>4s}  " + '  '.join((f'{e[-4:]:>16s}' for e in design_TARGET_EPOCHS)))
    for c in cells:
        parts = []
        for e in design_TARGET_EPOCHS:
            d = c['epochs'][e]
            n_sep = d['labels']['sepsis3']['n_events']
            parts.append(f"{d['n_common_cohort']:>7,} sep={n_sep:<5,}")
        print(f"{c['baseline_rule']:14s} {c['prediction_hour']:>4.0f} {c['horizon']:>4.0f}  " + '  '.join(parts))
    print('\nshare of onsets falling before the prediction hour (2014-16):')
    for c in cells:
        if c['horizon'] != 48.0:
            continue
        d = c['epochs']['2014 - 2016']['labels']
        print(f"  {c['baseline_rule']:14s} T={c['prediction_hour']:.0f}  " + '  '.join((f"{k}={v['frac_onsets_before_prediction']:.2f}" for k, v in d.items())))
    print(f'\n-> {path}   [{time.time() - t0:.0f}s]')

final_TARGET_EPOCHS = ['2014 - 2016', '2017 - 2019', '2020 - 2022']

def cohens_kappa(a: pl.Series, b: pl.Series) -> float:
    a, b = (a.cast(pl.Int8), b.cast(pl.Int8))
    n = a.len()
    po = float((a == b).mean())
    pa, pb = (float(a.mean()), float(b.mean()))
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 4) if n and pe < 1 else float('nan')

def final_main() -> None:
    t0 = time.time()
    cfg = load_config('labels/primary.yaml')
    task, sofa_cfg = (cfg['task'], cfg['sofa'])
    T, H, MAX = (task['prediction_hour'], task['horizon_hours'], task['max_hours'])
    min_cell = int(paths()['min_cell_count'])
    con = connect()
    build_cohort(con)
    register_source(con, 'mimic_iv', ['labevents', 'chartevents', 'outputevents', 'inputevents', 'prescriptions', 'microbiologyevents'])
    cohort = con.execute('SELECT stay_id, epoch FROM cohort').pl()
    stays = con.execute(f'SELECT stay_id, least(los_hours, {MAX}) AS max_hours FROM cohort').pl()
    w = weights(con)
    con.register('weights_tmp', w.to_arrow())
    creatinine = lab(con, 'creatinine', MAX)
    urine = urine_output(con, MAX)
    abx, cultures = (antibiotic_times(con, max_hours=MAX), culture_times(con, max_hours=MAX))
    sofa = hourly_sofa(SofaInputs(pao2=lab(con, 'pao2', MAX), fio2=chart(con, 'fio2', MAX), ventilated=ventilated_hours(con, MAX), platelets=lab(con, 'platelets', MAX), bilirubin=lab(con, 'bilirubin_total', MAX), mean_arterial_pressure=chart(con, 'map', MAX), gcs_total=gcs_total(con, MAX), creatinine=creatinine.filter(pl.col('hours') >= 0), urine=urine, vasopressors=vasopressor_rates(con, MAX)), stays)
    suspicion = suspicion_of_infection(abx, cultures)
    print(f'inputs ready ({time.time() - t0:.0f}s)')

    def build_onsets(baseline_rule: str) -> dict[str, pl.DataFrame]:
        return {'aki_creatinine': first_onset(creatinine_stage(creatinine)), 'sofa_dysfunction': organ_dysfunction_onset(sofa, prediction_hour=T, increase=sofa_cfg['increase'], baseline_rule=baseline_rule), 'aki_urine': first_onset(urine_output_stage(urine, w.select('stay_id', 'weight_kg'))), 'sepsis3': sepsis3_onset(sofa, suspicion, prediction_hour=T, increase=sofa_cfg['increase'], baseline_rule=baseline_rule)}

    def build_prevalent(onsets: dict[str, pl.DataFrame], rule: str) -> dict[str, pl.DataFrame]:
        return {'aki_creatinine': prevalent_from_onsets(onsets['aki_creatinine'], T), 'aki_urine': prevalent_from_onsets(onsets['aki_urine'], T), 'sofa_dysfunction': prevalent_dysfunction(sofa, prediction_hour=T, increase=sofa_cfg['increase'], rule=rule), 'sepsis3': prevalent_sepsis3(sofa, suspicion, prediction_hour=T, increase=sofa_cfg['increase'], rule=rule)}
    onsets = build_onsets(sofa_cfg['baseline_rule'])
    rule_comparison = []
    for rule in ('none', 'sofa_at_threshold', 'rise_before'):
        _, summ = common_cohort(cohort, build_prevalent(onsets, rule))
        rule_comparison.append({'prevalence_rule': rule, 'by_epoch': summ})
        sizes = ' '.join((f"{r['n_common_cohort']:>6,}" for r in summ))
        print(f'  prevalence_rule={rule:18s} common cohort: {sizes}')
    common, common_summary = common_cohort(cohort, build_prevalent(onsets, sofa_cfg['prevalence_rule']))
    print(f"chosen rule {sofa_cfg['prevalence_rule']!r}: common cohort {common.height:,} stays")
    labelled = common
    for name, o in onsets.items():
        labelled = labelled.join(o.select('stay_id', pl.col('onset_hours').alias(f'_h_{name}')), on='stay_id', how='left').with_columns(**{name: ((pl.col(f'_h_{name}') > T) & (pl.col(f'_h_{name}') <= H)).fill_null(False).cast(pl.Int8)})
    labelled = labelled.select(['stay_id', 'epoch', *onsets])
    prevalence = []
    for _k, g in sorted(labelled.group_by('epoch'), key=lambda kv: kv[0]):
        rec = {'epoch': g['epoch'][0], 'n': g.height}
        for name in onsets:
            n_pos = int(g[name].sum())
            rec[f'n_{name}'] = None if 0 < n_pos < min_cell else n_pos
            rec[f'prev_{name}'] = round(n_pos / g.height, 4)
        prevalence.append(rec)
    kappas = []
    for _k, g in sorted(labelled.group_by('epoch'), key=lambda kv: kv[0]):
        names = list(onsets)
        rec = {'epoch': g['epoch'][0]}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                rec[f'{a}__{b}'] = cohens_kappa(g[a], g[b])
        kappas.append(rec)
    action_rates = cohort.join(abx.filter(pl.col('hours').is_between(0, H)).select('stay_id').unique().with_columns(any_abx=pl.lit(1)), on='stay_id', how='left').join(cultures.filter(pl.col('hours').is_between(0, H)).select('stay_id').unique().with_columns(any_culture=pl.lit(1)), on='stay_id', how='left').join(suspicion.select('stay_id').unique().with_columns(any_suspicion=pl.lit(1)), on='stay_id', how='left').group_by('epoch').agg(n=pl.len(), frac_antibiotic=pl.col('any_abx').fill_null(0).mean().round(4), frac_culture=pl.col('any_culture').fill_null(0).mean().round(4), frac_suspicion=pl.col('any_suspicion').fill_null(0).mean().round(4)).sort('epoch')
    zero_onsets = build_onsets('zero')
    zero_prev = []
    for _k, g in sorted(cohort.group_by('epoch'), key=lambda kv: kv[0]):
        rec = {'epoch': g['epoch'][0], 'n': g.height}
        for name in ('sofa_dysfunction', 'sepsis3'):
            ids = set(zero_onsets[name].filter(pl.col('onset_hours') <= H)['stay_id'])
            rec[f'prev_{name}_any_within_48h'] = round(float(g['stay_id'].is_in(ids).mean()), 4)
        rec['prev_aki_creatinine_any_within_48h'] = round(float(g['stay_id'].is_in(set(zero_onsets['aki_creatinine'].filter(pl.col('onset_hours') <= H)['stay_id'])).mean()), 4)
        zero_prev.append(rec)
    report = {'design': cfg, 'prevalence_rule_comparison': rule_comparison, 'common_cohort': common_summary, 'prevalence_on_common_cohort': prevalence, 'cohens_kappa': kappas, 'clinician_action_rates': action_rates.to_dicts(), 'mimic_code_convention_prevalence': zero_prev}
    out = PROJECT_ROOT / 'results' / 'recon' / 'final_recon.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    by_epoch = {r['epoch']: r for r in prevalence}
    template = load_config('experiments/power.yaml')
    template['placeholder'] = False
    template['epoch_sizes'] = [by_epoch[e]['n'] for e in final_TARGET_EPOCHS]
    for contrast in template['contrasts']:
        for role, label in (('a', contrast['label_a']['name']), ('b', contrast['label_b']['name'])):
            key = {'sepsis3': 'sepsis3', 'sofa_dysfunction': 'sofa_dysfunction', 'aki_urine': 'aki_urine', 'aki_creatinine': 'aki_creatinine'}[label]
            contrast[f'label_{role}']['prevalence'] = round(sum((by_epoch[e][f'prev_{key}'] for e in final_TARGET_EPOCHS)) / len(final_TARGET_EPOCHS), 4)
    template['_note'] = 'epoch_sizes and prevalences measured on the common cohort at the frozen design; oracle_auroc values remain assumptions, as no model has been fitted'
    real = CONFIG_ROOT / 'experiments' / 'power_real.yaml'
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_text(yaml.safe_dump(template, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print('\ncommon cohort and prevalence on it:')
    hdr = '  '.join((f'{n[:13]:>16s}' for n in onsets))
    print(f"{'epoch':12s} {'n':>7s}  {hdr}")
    for r in prevalence:
        print(f"{r['epoch']:12s} {r['n']:>7,}  " + '  '.join((f"{r[f'prev_{n}']:>16.3f}" for n in onsets)))
    print("\nCohen's kappa between labels:")
    for r in kappas:
        pairs = {k: v for k, v in r.items() if k != 'epoch'}
        print(f"  {r['epoch']:12s} " + '  '.join((f"{k.split('__')[0][:4]}/{k.split('__')[1][:4]}={v:.2f}" for k, v in pairs.items())))
    print('\nclinician actions in the first 48 h:')
    print(action_rates)
    print('\nmimic-code convention (baseline 0), any event within 48 h:')
    for r in zero_prev:
        print(f"  {r['epoch']:12s} sepsis3={r['prev_sepsis3_any_within_48h']:.3f}  sofa={r['prev_sofa_dysfunction_any_within_48h']:.3f}  aki_cr={r['prev_aki_creatinine_any_within_48h']:.3f}")
    print(f'\n-> {out}\n-> {real}   [{time.time() - t0:.0f}s]')

def cultures_main() -> None:
    con = connect()
    build_cohort(con)
    register_source(con, 'mimic_iv', ['microbiologyevents', 'poe', 'prescriptions'])
    order_types = con.execute('\n        SELECT order_type, count(*) AS n\n        FROM mimic_iv_poe GROUP BY 1 ORDER BY n DESC LIMIT 20\n        ').fetchall()
    print('poe order_type inventory:')
    for t, n in order_types:
        print(f'  {str(t):28s} {n:>12,}')
    rows = con.execute("\n        WITH micro AS (\n            SELECT DISTINCT c.stay_id\n            FROM mimic_iv_microbiologyevents m\n            JOIN cohort c ON c.hadm_id = m.hadm_id\n            WHERE coalesce(m.charttime, m.chartdate) >= c.intime\n              AND coalesce(m.charttime, m.chartdate) <= c.intime + INTERVAL 48 HOUR\n        ), micro_any AS (\n            SELECT DISTINCT c.stay_id\n            FROM mimic_iv_microbiologyevents m\n            JOIN cohort c ON c.hadm_id = m.hadm_id\n        ), orders AS (\n            SELECT DISTINCT c.stay_id\n            FROM mimic_iv_poe p\n            JOIN cohort c ON c.hadm_id = p.hadm_id\n            WHERE lower(p.order_type) LIKE '%microbio%'\n              AND TRY_CAST(p.ordertime AS TIMESTAMP) >= c.intime\n              AND TRY_CAST(p.ordertime AS TIMESTAMP) <= c.intime + INTERVAL 48 HOUR\n        ), lab_orders AS (\n            SELECT DISTINCT c.stay_id\n            FROM mimic_iv_poe p\n            JOIN cohort c ON c.hadm_id = p.hadm_id\n            WHERE lower(p.order_type) LIKE '%lab%'\n              AND TRY_CAST(p.ordertime AS TIMESTAMP) >= c.intime\n              AND TRY_CAST(p.ordertime AS TIMESTAMP) <= c.intime + INTERVAL 48 HOUR\n        )\n        SELECT c.epoch,\n               count(*)                                        AS n_stays,\n               avg(CASE WHEN m.stay_id  IS NOT NULL THEN 1.0 ELSE 0 END) AS frac_specimen_48h,\n               avg(CASE WHEN ma.stay_id IS NOT NULL THEN 1.0 ELSE 0 END) AS frac_specimen_anytime,\n               avg(CASE WHEN o.stay_id  IS NOT NULL THEN 1.0 ELSE 0 END) AS frac_micro_order_48h,\n               avg(CASE WHEN lo.stay_id IS NOT NULL THEN 1.0 ELSE 0 END) AS frac_lab_order_48h\n        FROM cohort c\n        LEFT JOIN micro m       USING (stay_id)\n        LEFT JOIN micro_any ma  USING (stay_id)\n        LEFT JOIN orders o      USING (stay_id)\n        LEFT JOIN lab_orders lo USING (stay_id)\n        GROUP BY c.epoch ORDER BY c.epoch\n        ").fetchall()
    cols = ['epoch', 'n_stays', 'frac_specimen_48h', 'frac_specimen_anytime', 'frac_micro_order_48h', 'frac_lab_order_48h']
    table = [dict(zip(cols, r, strict=True)) for r in rows]
    timing = con.execute('\n        SELECT c.epoch,\n               count(*) AS n_specimens,\n               avg(CASE WHEN m.charttime IS NULL THEN 1.0 ELSE 0 END) AS frac_no_charttime\n        FROM mimic_iv_microbiologyevents m\n        JOIN cohort c ON c.hadm_id = m.hadm_id\n        GROUP BY c.epoch ORDER BY c.epoch\n        ').fetchall()
    print(f"\n{'epoch':12s} {'n':>7s} {'spec48h':>9s} {'specEver':>9s} {'microOrd':>9s} {'labOrd':>8s}")
    for r in table:
        print(f"{r['epoch']:12s} {r['n_stays']:>7,} {r['frac_specimen_48h']:>9.3f} {r['frac_specimen_anytime']:>9.3f} {r['frac_micro_order_48h']:>9.3f} {r['frac_lab_order_48h']:>8.3f}")
    print(f"\n{'epoch':12s} {'specimens':>11s} {'no charttime':>13s}")
    for e, n, f in timing:
        print(f'{e:12s} {n:>11,} {f:>13.3f}')
    subtypes = con.execute("\n        SELECT order_subtype, count(*) AS n\n        FROM mimic_iv_poe\n        WHERE lower(order_type) LIKE '%lab%'\n        GROUP BY 1 ORDER BY n DESC LIMIT 25\n        ").fetchall()
    print("\npoe order_subtype under 'Lab':")
    for s, n in subtypes:
        print(f'  {str(s):34s} {n:>12,}')
    culture_orders = con.execute("\n        WITH co AS (\n            SELECT DISTINCT c.stay_id\n            FROM mimic_iv_poe p\n            JOIN cohort c ON c.hadm_id = p.hadm_id\n            WHERE (lower(p.order_subtype) LIKE '%culture%'\n                   OR lower(p.order_subtype) LIKE '%micro%'\n                   OR lower(p.order_subtype) LIKE '%blood cx%')\n              AND TRY_CAST(p.ordertime AS TIMESTAMP) >= c.intime\n              AND TRY_CAST(p.ordertime AS TIMESTAMP) <= c.intime + INTERVAL 48 HOUR\n        )\n        SELECT c.epoch, count(*) AS n_stays,\n               avg(CASE WHEN co.stay_id IS NOT NULL THEN 1.0 ELSE 0 END) AS frac_culture_order\n        FROM cohort c LEFT JOIN co USING (stay_id)\n        GROUP BY c.epoch ORDER BY c.epoch\n        ").fetchall()
    per_stay = con.execute('\n        SELECT c.epoch, count(m.hadm_id)::DOUBLE / count(DISTINCT c.stay_id)\n        FROM cohort c\n        LEFT JOIN mimic_iv_microbiologyevents m ON m.hadm_id = c.hadm_id\n        GROUP BY c.epoch ORDER BY c.epoch\n        ').fetchall()
    print(f"\n{'epoch':12s} {'cultOrder48h':>13s} {'specimens/stay':>15s}")
    for (e, _n, f), (_e2, pps) in zip(culture_orders, per_stay, strict=True):
        print(f'{e:12s} {f:>13.3f} {pps:>15.1f}')
    out = {'poe_order_types': [{'order_type': str(t), 'n': int(n)} for t, n in order_types], 'poe_lab_subtypes': [{'order_subtype': str(s), 'n': int(n)} for s, n in subtypes], 'culture_orders_by_epoch': [{'epoch': e, 'n_stays': int(n), 'frac_culture_order_48h': round(float(f), 4)} for e, n, f in culture_orders], 'specimens_per_stay': [{'epoch': e, 'specimens_per_stay': round(float(p), 2)} for e, p in per_stay], 'coverage_by_epoch': table, 'specimen_timing': [{'epoch': e, 'n_specimens': int(n), 'frac_no_charttime': round(float(f), 4)} for e, n, f in timing]}
    path = PROJECT_ROOT / 'results' / 'recon' / 'culture_check.json'
    path.write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')
    print(f'\n-> {path}')

STERILE_SITES = ['BLOOD CULTURE', 'BLOOD CULTURE - NEONATE', 'BLOOD CULTURE (POST-MORTEM)', 'CSF;SPINAL FLUID', 'PERITONEAL FLUID', 'PLEURAL FLUID', 'JOINT FLUID', 'FLUID RECEIVED IN BLOOD CULTURE BOTTLES', 'BILE', 'ABSCESS', 'TISSUE', 'BONE MARROW', 'PERICARDIAL FLUID', 'DIALYSIS FLUID']

def suspicion_main() -> None:
    con = connect()
    build_cohort(con)
    register_source(con, 'mimic_iv', ['prescriptions', 'microbiologyevents'])
    cfg = load_config('concepts/mimic_iv.yaml')['antibiotics']
    likes = ' OR '.join((f"lower(p.drug) LIKE '%{n}%'" for n in cfg['names']))
    routes = ', '.join((f"'{r}'" for r in cfg['routes_excluded']))
    print('antibiotic routes among matched prescriptions (top 15):')
    for route, n in con.execute(f"\n        SELECT upper(coalesce(p.route, 'NULL')) AS route, count(*) AS n\n        FROM mimic_iv_prescriptions p JOIN cohort c ON c.hadm_id = p.hadm_id\n        WHERE ({likes}) GROUP BY 1 ORDER BY n DESC LIMIT 15\n        ").fetchall():
        print(f'  {route:22s} {n:>10,}')
    print('\nmost frequent matched drugs (top 15):')
    for drug, n in con.execute(f'\n        SELECT lower(p.drug) AS drug, count(*) AS n\n        FROM mimic_iv_prescriptions p JOIN cohort c ON c.hadm_id = p.hadm_id\n        WHERE ({likes}) GROUP BY 1 ORDER BY n DESC LIMIT 15\n        ').fetchall():
        print(f'  {drug:34s} {n:>10,}')
    print('\nspecimen types (top 15):')
    for spec, n in con.execute('\n        SELECT upper(m.spec_type_desc) AS s, count(*) AS n\n        FROM mimic_iv_microbiologyevents m JOIN cohort c ON c.hadm_id = m.hadm_id\n        GROUP BY 1 ORDER BY n DESC LIMIT 15\n        ').fetchall():
        print(f'  {str(spec):38s} {n:>10,}')
    sterile = ', '.join((f"'{s}'" for s in STERILE_SITES))
    iv = "upper(p.route) IN ('IV','IV DRIP','IVPCA','IM','IV BOLUS')"
    not_surveillance = "upper(m.spec_type_desc) NOT LIKE '%SCREEN%' AND upper(m.spec_type_desc) NOT LIKE '%MRSA%' AND upper(m.spec_type_desc) NOT LIKE '%VRE%' AND upper(m.spec_type_desc) NOT LIKE '%SURVEILLANCE%' AND upper(m.spec_type_desc) <> 'STAPH AUREUS SWAB'"
    variants = {'current': ('1=1', '1=1'), 'iv_only': (iv, '1=1'), 'sterile_only': ('1=1', f'upper(m.spec_type_desc) IN ({sterile})'), 'iv_and_sterile': (iv, f'upper(m.spec_type_desc) IN ({sterile})'), 'no_surveillance': ('1=1', not_surveillance), 'iv_and_no_surveillance': (iv, not_surveillance)}
    rows = []
    for name, (abx_filter, cult_filter) in variants.items():
        res = con.execute(f"\n            WITH abx AS (\n                SELECT DISTINCT c.stay_id,\n                       date_diff('second', c.intime, p.starttime)/3600.0 AS h\n                FROM mimic_iv_prescriptions p JOIN cohort c ON c.hadm_id = p.hadm_id\n                WHERE ({likes})\n                  AND (p.route IS NULL OR upper(p.route) NOT IN ({routes}))\n                  AND {abx_filter}\n                  AND p.starttime BETWEEN c.intime - INTERVAL 24 HOUR\n                                      AND c.intime + INTERVAL 96 HOUR\n            ), cul AS (\n                SELECT DISTINCT c.stay_id,\n                       date_diff('second', c.intime,\n                                 coalesce(m.charttime, m.chartdate))/3600.0 AS h\n                FROM mimic_iv_microbiologyevents m JOIN cohort c ON c.hadm_id = m.hadm_id\n                WHERE {cult_filter}\n                  AND coalesce(m.charttime, m.chartdate)\n                      BETWEEN c.intime - INTERVAL 24 HOUR AND c.intime + INTERVAL 96 HOUR\n            ), paired AS (\n                SELECT DISTINCT a.stay_id\n                FROM abx a JOIN cul u USING (stay_id)\n                WHERE (u.h - a.h BETWEEN 0 AND 24) OR (a.h - u.h BETWEEN 0 AND 72)\n            )\n            SELECT c.epoch, count(*) AS n_stays,\n                   count(pr.stay_id) AS n_suspicion\n            FROM cohort c LEFT JOIN paired pr USING (stay_id)\n            GROUP BY c.epoch ORDER BY c.epoch\n            ").fetchall()
        total_n = sum((r[1] for r in res))
        total_s = sum((r[2] for r in res))
        rows.append({'variant': name, 'n_cohort': total_n, 'n_suspicion': total_s, 'frac_overall': round(total_s / total_n, 4), 'by_epoch': [{'epoch': e, 'n_stays': int(n), 'n_suspicion': int(s), 'frac': round(s / n, 4)} for e, n, s in res]})
    print(f"\n{'variant':16s} {'suspicion':>10s} {'frac':>7s}   by epoch")
    for r in rows:
        fr = ' '.join((f"{d['frac']:.2f}" for d in r['by_epoch']))
        print(f"{r['variant']:16s} {r['n_suspicion']:>10,} {r['frac_overall']:>7.3f}   {fr}")
    out = PROJECT_ROOT / 'results' / 'recon' / 'suspicion_calibration.json'
    out.write_text(json.dumps({'variants': rows}, indent=2), encoding='utf-8')
    print(f'\n-> {out}')

STAGES = {
    'power': power_main,
    'aki': aki_main,
    'ventilation': ventilation_main,
    'labels': labels_main,
    'design': design_main,
    'final': final_main,
    'cultures': cultures_main,
    'suspicion': suspicion_main,
}

if __name__ == '__main__':
    argv = sys.argv[1:]
    chosen = [argv[0]] if argv and argv[0] in STAGES else list(STAGES)
    rest = argv[1:] if argv and argv[0] in STAGES else argv
    for name in chosen:
        sys.argv = [name, *rest]
        print('== ' + name + ' ==')
        STAGES[name]()
