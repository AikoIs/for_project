from __future__ import annotations
import copy
import sys
import yaml
from pathlib import Path
import argparse
import duckdb
import json
import numpy as np
import platform
import sys
import time

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




def source_root(source):
    return Path(PATHS["sources"][source]["csv_root"])


def parquet_dir(source):
    return Path(PATHS["parquet_root"]) / source


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

EPOCHS = ['2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019', '2020 - 2022']

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

def attrition(con: duckdb.DuckDBPyConnection) -> list[dict]:
    register_source(con, 'mimic_iv', ['patients', 'admissions', 'icustays'])
    steps = [('all ICU stays', 'SELECT count(*) FROM mimic_iv_icustays'), ('distinct patients with an ICU stay', 'SELECT count(DISTINCT subject_id) FROM mimic_iv_icustays'), ('first ICU stay per patient', 'SELECT count(*) FROM (SELECT subject_id, min(intime)\n            FROM mimic_iv_icustays GROUP BY subject_id)'), ('... and LOS >= 12 h', f'\n            SELECT count(*) FROM (\n              SELECT subject_id, los * 24.0 AS h,\n                     ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) rk\n              FROM mimic_iv_icustays) WHERE rk = 1 AND h >= {MIN_LOS_HOURS}'), ('... and age >= 18 (final cohort)', 'SELECT count(*) FROM cohort')]
    return [{'step': name, 'n': con.execute(q).fetchone()[0]} for name, q in steps]


def _header_columns(con, csv_path: Path) -> set[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_csv('{csv_path.as_posix()}', header=true, delim=',', sample_size=1000)").fetchall()
    return {r[0] for r in rows}

def _read_csv_expr(con, csv_path: Path, spec: dict, defaults: dict) -> tuple[str, list[str]]:
    opts = [f"'{csv_path.as_posix()}'", 'header=true', "delim=','", 'quote=\'"\'', 'escape=\'"\'', f"sample_size={int(defaults['sample_size'])}", 'parallel=true']
    dropped: list[str] = []
    if spec.get('all_varchar'):
        opts.append('all_varchar=true')
    elif spec.get('types'):
        present = _header_columns(con, csv_path)
        types = {k: v for k, v in spec['types'].items() if k in present}
        dropped = sorted(set(spec['types']) - present)
        if types:
            pairs = ', '.join((f"'{k}': '{v}'" for k, v in types.items()))
            opts.append(f'types={{{pairs}}}')
    return (f"read_csv({', '.join(opts)})", dropped)

def convert_source(source: str, cfg: dict, force: bool) -> list[dict]:
    defaults = cfg['defaults']
    tables = cfg['sources'][source]['tables']
    csv_root = source_root(source)
    out_dir = parquet_dir(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = connect()
    records = []
    for rel_name, spec in tables.items():
        name = rel_name.split('/')[-1]
        out_path = out_dir / f'{name}.parquet'
        if spec.get('skip'):
            print(f"[skip] {source}/{name}: {spec.get('reason', '')}")
            records.append({'source': source, 'table': name, 'status': 'skipped', 'reason': spec.get('reason', '')})
            continue
        csv_path = csv_root / f'{rel_name}.csv'
        if not csv_path.exists():
            csv_path = csv_root / f'{rel_name}.csv.gz'
        if not csv_path.exists():
            print(f'[MISSING] {source}/{rel_name}')
            records.append({'source': source, 'table': name, 'status': 'missing', 'expected': str(csv_path)})
            continue
        if out_path.exists() and (not force):
            n = con.execute(f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
            print(f'[have] {source}/{name}: {n:,} rows')
            records.append({'source': source, 'table': name, 'status': 'cached', 'rows': n, 'parquet_bytes': out_path.stat().st_size})
            continue
        expr, dropped = _read_csv_expr(con, csv_path, spec, defaults)
        if dropped:
            print(f'[note] {source}/{name}: type pins for absent columns ignored: {dropped}')
        t0 = time.time()
        con.execute(f"COPY (SELECT * FROM {expr}) TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION {defaults['compression'].upper()}, ROW_GROUP_SIZE {int(defaults['row_group_size'])})")
        elapsed = time.time() - t0
        n = con.execute(f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
        rec = {'source': source, 'table': name, 'status': 'converted', 'rows': n, 'csv_bytes': csv_path.stat().st_size, 'parquet_bytes': out_path.stat().st_size, 'seconds': round(elapsed, 1)}
        records.append(rec)
        print(f"[ok]   {source}/{name}: {n:,} rows, {rec['csv_bytes'] / 1000000000.0:.2f} GB -> {rec['parquet_bytes'] / 1000000000.0:.2f} GB, {elapsed:.0f}s")
    con.close()
    return records

def convert_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='all', choices=['all', 'mimic_iv', 'eicu'])
    ap.add_argument('--force', action='store_true', help='re-convert tables that already exist')
    args = ap.parse_args()
    cfg = load_config('datasets/conversion.yaml')
    sources = [s for s in cfg['sources'] if s in PATHS['sources']] if args.source == 'all' else [args.source]
    manifest_path = PROJECT_ROOT / 'results' / 'recon' / 'conversion_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[dict]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for source in sources:
        manifest[source] = convert_source(source, cfg, args.force)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nmanifest -> {manifest_path}')

def check_torch() -> dict:
    import torch
    info = {'version': torch.__version__, 'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}
    if not info['cuda_available']:
        info['status'] = 'FAIL: no CUDA device visible'
        return info
    info['device_name'] = torch.cuda.get_device_name(0)
    info['capability'] = '.'.join(map(str, torch.cuda.get_device_capability(0)))
    info['arch_list'] = torch.cuda.get_arch_list()
    info['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1000000000.0, 1)
    try:
        dev = torch.device('cuda')
        a = torch.randn(4096, 4096, device=dev)
        t0 = time.time()
        for _ in range(10):
            a = a @ a.T / 4096.0
        torch.cuda.synchronize()
        info['matmul_10x4096_seconds'] = round(time.time() - t0, 3)
        gru = torch.nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True).to(dev)
        x = torch.randn(256, 24, 32, device=dev)
        out, _ = gru(x)
        out.sum().backward()
        torch.cuda.synchronize()
        info['gru_forward_backward'] = 'ok'
        info['status'] = 'OK'
    except Exception as exc:
        info['status'] = f'FAIL: {type(exc).__name__}: {exc}'
    return info

def check_xgboost() -> dict:
    import xgboost as xgb
    info = {'version': xgb.__version__}
    try:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20000, 40))
        y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.5, size=20000) > 0).astype(int)
        dtrain = xgb.DMatrix(x, label=y)
        t0 = time.time()
        booster = xgb.train({'device': 'cuda', 'tree_method': 'hist', 'objective': 'binary:logistic', 'max_depth': 4, 'eta': 0.2}, dtrain, num_boost_round=50)
        pred = booster.predict(dtrain)
        info['gpu_train_seconds'] = round(time.time() - t0, 3)
        info['pred_mean'] = round(float(pred.mean()), 4)
        info['status'] = 'OK'
    except Exception as exc:
        info['status'] = f'FAIL: {type(exc).__name__}: {exc}'
    return info

def check_duckdb() -> dict:
    import duckdb
    con = duckdb.connect()
    n = con.execute('SELECT count(*) FROM range(1000000)').fetchone()[0]
    return {'version': duckdb.__version__, 'smoke_rows': n, 'status': 'OK'}

def check_env_main() -> None:
    report = {'python': sys.version.split()[0], 'platform': platform.platform(), 'torch': check_torch(), 'xgboost': check_xgboost(), 'duckdb': check_duckdb()}
    out = PROJECT_ROOT / 'results' / 'recon' / 'environment.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    failures = [k for k in ('torch', 'xgboost', 'duckdb') if not str(report[k].get('status', '')).startswith('OK')]
    if failures:
        print(f'\nFAILED: {failures}')
        sys.exit(1)
    print('\nall green')

LAB_PATTERNS = {'creatinine': ['creatinine'], 'bilirubin': ['bilirubin'], 'platelets': ['platelet'], 'lactate': ['lactate', 'lactic'], 'pao2': ['po2', 'oxygen'], 'bicarbonate': ['bicarbonate'], 'wbc': ['white blood cell', 'leukocyte'], 'sodium': ['sodium'], 'potassium': ['potassium'], 'hemoglobin': ['hemoglobin'], 'bun': ['urea nitrogen'], 'glucose': ['glucose']}

ITEM_PATTERNS = {'heart_rate': ['heart rate'], 'sbp': ['systolic', 'arterial blood pressure systolic', 'non invasive blood pressure systolic'], 'map': ['mean arterial', 'blood pressure mean', 'arterial blood pressure mean'], 'resp_rate': ['respiratory rate'], 'temperature': ['temperature'], 'spo2': ['o2 saturation', 'spo2'], 'fio2': ['fio2', 'inspired o2'], 'gcs': ['gcs', 'glasgow'], 'urine_output': ['urine', 'foley', 'void'], 'weight': ['weight'], 'height': ['height'], 'vent_mode': ['ventilator mode', 'o2 delivery device'], 'vasopressor': ['norepinephrine', 'epinephrine', 'dopamine', 'dobutamine', 'vasopressin', 'phenylephrine']}

def _views(con: duckdb.DuckDBPyConnection, source: str, names: dict[str, str]) -> None:
    root = parquet_dir(source)
    for view, fname in names.items():
        path = root / f'{fname}.parquet'
        if path.exists():
            con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{path.as_posix()}')")

def search(con, table: str, label_col: str, extra_cols: list[str], patterns: dict) -> dict:
    out = {}
    cols = ', '.join(['itemid', label_col, *extra_cols])
    for concept, needles in patterns.items():
        clause = ' OR '.join((f"lower({label_col}) LIKE '%{n.lower()}%'" for n in needles))
        rows = con.execute(f'SELECT {cols} FROM {table} WHERE {clause} ORDER BY itemid').fetchall()
        out[concept] = [dict(zip(['itemid', 'label', *extra_cols], r, strict=True)) for r in rows]
    return out

def dictionaries_main() -> None:
    con = duckdb.connect()
    report: dict = {}
    _views(con, 'mimic_iv', {'iv_dlab': 'd_labitems', 'iv_ditems': 'd_items'})
    report['mimic_iv_labs'] = search(con, 'iv_dlab', 'label', ['fluid', 'category'], LAB_PATTERNS)
    report['mimic_iv_items'] = search(con, 'iv_ditems', 'label', ['linksto', 'category', 'unitname', 'param_type'], ITEM_PATTERNS)
    out = PROJECT_ROOT / 'results' / 'recon' / 'concept_candidates.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    for block, concepts in report.items():
        print(f'\n===== {block} =====')
        for concept, rows in concepts.items():
            print(f'  {concept:14s} {len(rows):4d} candidates')
    print(f'\n-> {out}')

MIN_CELL = None

def _suppress(n: int) -> int | None:
    return None if n > 0 and n < MIN_CELL else n

def cohort_by_epoch(con) -> dict:
    rows = con.execute("\n        SELECT epoch,\n               count(*)                                   AS n_stays,\n               count(DISTINCT subject_id)                 AS n_patients,\n               median(age)                                AS age_median,\n               quantile_cont(age, 0.25)                   AS age_q25,\n               quantile_cont(age, 0.75)                   AS age_q75,\n               avg(CASE WHEN gender = 'F' THEN 1.0 ELSE 0.0 END)   AS frac_female,\n               median(los_hours)                          AS los_median,\n               quantile_cont(los_hours, 0.25)             AS los_q25,\n               quantile_cont(los_hours, 0.75)             AS los_q75,\n               avg(hospital_expire_flag::DOUBLE)          AS frac_hosp_mortality,\n               median(anchor_offset_years)                AS anchor_offset_median,\n               avg(CASE WHEN abs(anchor_offset_years) <= 1 THEN 1.0 ELSE 0.0 END)\n                                                          AS frac_within_1y_of_anchor\n        FROM cohort GROUP BY epoch ORDER BY epoch\n        ").fetchall()
    cols = [d[0] for d in con.description]
    out = []
    for r in rows:
        rec = dict(zip(cols, r, strict=True))
        rec['n_stays'] = _suppress(rec['n_stays'])
        rec['n_patients'] = _suppress(rec['n_patients'])
        out.append(rec)
    return {'by_epoch': out, 'total': con.execute('SELECT count(*) FROM cohort').fetchone()[0]}

def _dictionary_labels(con, source: str) -> dict[int, str]:
    register_source(con, source, ['d_items', 'd_labitems'])
    labels = {}
    for view, id_col, label_col in (('mimic_iv_d_items', 'itemid', 'label'), ('mimic_iv_d_labitems', 'itemid', 'label')):
        for iid, lab in con.execute(f'SELECT {id_col}, {label_col} FROM {view}').fetchall():
            labels[int(iid)] = lab
    return labels

def coverage(con, table: str, id_col: str, time_col: str, stay_key: str, itemids: list[int], value_col: str | None) -> dict:
    ids = ', '.join((str(i) for i in itemids))
    if stay_key == 'stay_id':
        join = f'JOIN cohort c ON c.stay_id = e.stay_id'
    else:
        join = f'JOIN cohort c ON c.hadm_id = e.hadm_id'
    per_item = con.execute(f'\n        SELECT e.{id_col} AS itemid,\n               count(*) AS n_rows,\n               count(DISTINCT c.stay_id) AS n_stays\n        FROM {table} e {join}\n        WHERE e.{id_col} IN ({ids})\n          AND e.{time_col} BETWEEN c.intime - INTERVAL 7 DAY AND c.intime + INTERVAL 48 HOUR\n        GROUP BY 1 ORDER BY 1\n        ').fetchall()
    window = con.execute(f"\n        WITH measured AS (\n            SELECT DISTINCT c.stay_id, c.epoch\n            FROM {table} e {join}\n            WHERE e.{id_col} IN ({ids})\n              AND e.{time_col} >= c.intime\n              AND e.{time_col} <= c.intime + INTERVAL 6 HOUR\n              {('AND e.' + value_col + ' IS NOT NULL' if value_col else '')}\n        )\n        SELECT c.epoch,\n               count(*) AS n_stays,\n               count(m.stay_id) AS n_measured\n        FROM cohort c LEFT JOIN measured m USING (stay_id)\n        GROUP BY c.epoch ORDER BY c.epoch\n        ").fetchall()
    return {'per_itemid': [{'itemid': int(i), 'n_rows': int(r), 'n_stays': _suppress(int(s))} for i, r, s in per_item], 'window_0_6h': [{'epoch': e, 'n_stays': int(n), 'n_measured': _suppress(int(m)), 'frac_measured': round(m / n, 4) if n else None} for e, n, m in window]}

def concepts_main() -> None:
    global MIN_CELL
    MIN_CELL = int(paths()['min_cell_count'])
    cfg = load_config('concepts/mimic_iv.yaml')
    con = connect()
    build_cohort(con)
    report = {'attrition': attrition(con), 'cohort': cohort_by_epoch(con), 'epochs_expected': EPOCHS}
    print('attrition:')
    for step in report['attrition']:
        print(f"  {step['step']:40s} {step['n']:>8,}")
    print('\ncohort by epoch:')
    for r in report['cohort']['by_epoch']:
        print(f"  {r['epoch']:12s} stays={r['n_stays']:>7,}  age={r['age_median']:.0f}  los_h={r['los_median']:.0f}  within1y={r['frac_within_1y_of_anchor']:.2f}")
    labels = _dictionary_labels(con, 'mimic_iv')
    register_source(con, 'mimic_iv', ['labevents', 'chartevents', 'outputevents', 'inputevents', 'procedureevents'])
    spec = [('labs', 'mimic_iv_labevents', 'itemid', 'charttime', 'hadm_id', 'valuenum'), ('chart', 'mimic_iv_chartevents', 'itemid', 'charttime', 'stay_id', 'valuenum'), ('output', 'mimic_iv_outputevents', 'itemid', 'charttime', 'stay_id', 'value'), ('input', 'mimic_iv_inputevents', 'itemid', 'starttime', 'stay_id', None), ('procedure', 'mimic_iv_procedureevents', 'itemid', 'starttime', 'stay_id', None)]
    concepts: dict = {}
    unknown: list = []
    empty: list = []
    for block, table, id_col, time_col, key, value_col in spec:
        if block not in cfg:
            continue
        print(f'\n--- {block} ---')
        concepts[block] = {}
        for name, entry in cfg[block].items():
            ids = list(entry['itemids']) + list(entry.get('subtract_itemids', []))
            for iid in ids:
                if iid not in labels:
                    unknown.append({'block': block, 'concept': name, 'itemid': iid})
            res = coverage(con, table, id_col, time_col, key, ids, value_col)
            seen = {d['itemid'] for d in res['per_itemid']}
            for iid in ids:
                if iid not in seen:
                    empty.append({'block': block, 'concept': name, 'itemid': iid, 'label': labels.get(iid)})
            for d in res['per_itemid']:
                d['label'] = labels.get(d['itemid'])
            concepts[block][name] = res
            fr = [w['frac_measured'] for w in res['window_0_6h']]
            print(f'  {name:18s} 0-6h coverage by epoch: ' + ' '.join((f'{f:.2f}' if f is not None else '  . ' for f in fr)))
    report['concepts'] = concepts
    report['itemids_not_in_dictionary'] = unknown
    report['itemids_with_no_rows_in_cohort'] = empty
    if unknown:
        print(f'\n!! {len(unknown)} itemid(s) absent from the dictionary: {unknown}')
    if empty:
        print(f'\n!! {len(empty)} itemid(s) present but with no rows in the cohort window:')
        for e in empty:
            print(f"   {e['block']}/{e['concept']}: {e['itemid']} ({e['label']})")
    out = PROJECT_ROOT / 'results' / 'recon' / 'concept_coverage.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(f'\n-> {out}')

STAGES = {
    'convert': convert_main,
    'check_env': check_env_main,
    'dictionaries': dictionaries_main,
    'concepts': concepts_main,
}

if __name__ == '__main__':
    argv = sys.argv[1:]
    chosen = [argv[0]] if argv and argv[0] in STAGES else list(STAGES)
    rest = argv[1:] if argv and argv[0] in STAGES else argv
    for name in chosen:
        sys.argv = [name, *rest]
        print('== ' + name + ' ==')
        STAGES[name]()
