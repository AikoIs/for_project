from __future__ import annotations
import copy
import sys
import yaml
from collections import defaultdict
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import csv
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statistics as st

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










FIGURE_DIR = PROJECT_ROOT / 'figures'

OKABE_ITO = {'black': '#000000', 'orange': '#E69F00', 'sky_blue': '#56B4E9', 'bluish_green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00', 'reddish_purple': '#CC79A7'}

LABEL_COLOUR = {'aki_creatinine': OKABE_ITO['blue'], 'sofa_dysfunction': OKABE_ITO['sky_blue'], 'aki_urine': OKABE_ITO['orange'], 'sepsis3': OKABE_ITO['vermillion']}

LABEL_DISPLAY = {'aki_creatinine': 'AKI (creatinine)', 'sofa_dysfunction': 'SOFA deterioration', 'aki_urine': 'AKI (urine output)', 'sepsis3': 'Sepsis-3'}

LABEL_SHORT = {'aki_creatinine': 'creatinine-AKI', 'sofa_dysfunction': 'SOFA deterioration', 'aki_urine': 'urine-AKI', 'sepsis3': 'Sepsis-3'}


CONTRAST_ACTION_LABEL = {'DiD_sepsis': 'sepsis3', 'DiD_aki': 'aki_urine'}

CONTRAST_DISPLAY = {'DiD_sepsis': 'DiD$_{\\mathrm{sepsis}}$', 'DiD_aki': 'DiD$_{\\mathrm{AKI}}$'}

SIZE_MARKER = {'full': 'o', 'matched': 's', 'event_matched': '^'}

SIZE_FILLED = {'full': True, 'matched': False, 'event_matched': False}

SIZE_DISPLAY = {'full': 'full', 'matched': 'matched', 'event_matched': 'event-matched'}

VERDICT_SHORT = {'practically equivalent': 'equivalent', 'inconclusive': 'inconclusive', 'distinguishable but trivial': 'trivial', 'meaningful drift asymmetry': 'meaningful', 'meaningful asymmetry in the opposite direction': 'meaningful, opposite sign'}


TEXT_WIDTH = 7.0

GREY = '#4D4D4D'

LIGHT_GREY = '#BFBFBF'

BAND_GREY = '#E6E6E6'

def use_style():
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.direction': 'out', 'ytick.direction': 'out', 'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.0, 'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def sesoi_band(ax, sesoi, vertical=True, label=None):
    span = ax.axvspan if vertical else ax.axhspan
    line = ax.axvline if vertical else ax.axhline
    span(-sesoi, sesoi, color=BAND_GREY, zorder=0, lw=0, label=label)
    line(0.0, color=GREY, lw=0.7, zorder=1)


def save(fig, stem):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ('.pdf', '.png'):
        path = FIGURE_DIR / (stem + suffix)
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    for path in written:
        try:
            print('  wrote ' + str(path.relative_to(PROJECT_ROOT)))
        except ValueError:
            print('  wrote ' + str(path))
    return written

def row(text, point=None, lo=None, hi=None, colour=None, marker='o', filled=True, starred=False, notes=(), heading=False, rule_above=False, muted=False):
    return {'text': text, 'point': point, 'lo': lo, 'hi': hi, 'colour': GREY if colour is None else colour, 'marker': marker, 'filled': filled, 'starred': starred, 'notes': notes, 'heading': heading, 'rule_above': rule_above, 'muted': muted}

def draw(ax, rows, sesoi=None, xlabel='', note_x=(), note_headers=(), xlim=None, heading_x=-0.6):
    sesoi_band(ax, sesoi)
    ys = list(range(len(rows)))
    for i in ys:
        r = rows[i]
        y = -i
        if r['rule_above']:
            ax.axhline(y + 0.5, color=LIGHT_GREY, lw=0.5, zorder=0)
        if r['heading']:
            continue
        ax.plot([r['lo'], r['hi']], [y, y], color=r['colour'], lw=1.3, solid_capstyle='butt', zorder=3)
        ax.plot([r['point']], [y], marker=r['marker'], ms=4.2, zorder=4, color=r['colour'], markerfacecolor=r['colour'] if r['filled'] else 'white', markeredgecolor=r['colour'], markeredgewidth=1.0)
    ax.set_yticks([-i for i in ys])
    ax.set_yticklabels(['' if r['heading'] else r['text'] for r in rows])
    ax.tick_params(axis='y', length=0)
    ticks = ax.get_yticklabels()
    for i in ys:
        if rows[i]['muted']:
            ticks[i].set_color(GREY)
            ticks[i].set_style('italic')
    for i in ys:
        r = rows[i]
        if r['heading']:
            ax.annotate(r['text'], xy=(heading_x, -i), xycoords=('axes fraction', 'data'), va='center', ha='left', fontweight='bold', annotation_clip=False)
    ax.set_ylim(-len(rows) + 0.4, 0.6)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlabel(xlabel)
    ax.spines['left'].set_visible(False)
    coords = ('axes fraction', 'data')
    for i in ys:
        r = rows[i]
        if r['heading']:
            continue
        for j in range(len(note_x)):
            ax.annotate(r['notes'][j], xy=(note_x[j], -i), xycoords=coords, va='center', ha='left', fontsize=7.2, annotation_clip=False)
        if r['starred']:
            ax.annotate('$\\bigstar$', xy=(note_x[0] - 0.075, -i), xycoords=coords, va='center', ha='left', fontsize=7.2, color=r['colour'], annotation_clip=False)
    for j in range(len(note_x)):
        ax.annotate(note_headers[j], xy=(note_x[j], 1.0), xycoords=coords, va='bottom', ha='left', fontsize=7.2, style='italic', color=GREY, annotation_clip=False)

def direction_key(ax, left='', right='', y=-0.19, line_gap=0.05):
    ax.annotate(left, xy=(0.0, y), xycoords='axes fraction', ha='left', va='top', fontsize=7, color=GREY, annotation_clip=False)
    ax.annotate(right, xy=(1.0, y - line_gap), xycoords='axes fraction', ha='right', va='top', fontsize=7, color=GREY, annotation_clip=False)

fig01_RESULTS = PROJECT_ROOT / 'results'

BOX = 'round,pad=0.012,rounding_size=0.02'

def fig01_epoch_short(epoch: str) -> str:
    first, last = [part.strip() for part in epoch.split('-')]
    return f'{first}–{last[2:]}'

def panel_label(ax, letter: str, text: str) -> None:
    ax.annotate(f'({letter})  {text}', xy=(0.0, 1.0), xycoords='axes fraction', ha='left', va='top', fontsize=8.5, fontweight='bold')

def blank(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

def box(ax, x, y, w, h, *, fc, ec, lw=0.8, zorder=2):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=BOX, fc=fc, ec=ec, lw=lw, mutation_aspect=0.5, zorder=zorder)
    ax.add_patch(patch)
    return patch

def panel_epochs(ax, recon, cfg) -> None:
    blank(ax)
    panel_label(ax, 'a', 'temporal axis (MIMIC-IV): train once, transfer forward')
    rows = recon['common_cohort']
    n = len(rows)
    width, gap = (0.165, 0.03)
    xs = [0.018 + i * (width + gap) for i in range(n)]
    centres = [x + width / 2 for x in xs]
    y, h = (0.36, 0.19)
    train = set(cfg['train_epochs'])
    for x, row in zip(xs, rows, strict=True):
        is_train = row['epoch'] in train
        box(ax, x, y, width, h, fc='#EDF2F7' if is_train else 'white', ec=GREY, lw=1.1 if is_train else 0.8)
        ax.text(x + width / 2, y + h * 0.62, fig01_epoch_short(row['epoch']), ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x + width / 2, y + h * 0.24, f"n = {row['n_common_cohort']:,}", ha='center', va='center', fontsize=7, color=GREY)
    n_train = sum((r['n_common_cohort'] for r in rows if r['epoch'] in train))
    ax.annotate('', xy=(xs[1] + width, y - 0.055), xytext=(xs[0], y - 0.055), arrowprops=dict(arrowstyle='-', color=GREY, lw=0.8))
    ax.text((centres[0] + centres[1]) / 2, y - 0.145, f'training era, pooled: n = {n_train:,}', ha='center', va='top', fontsize=7.5)
    source = (centres[0] + centres[1]) / 2
    panel_inches = TEXT_WIDTH * 0.97
    for centre in centres[len(train):]:
        rad = -0.4 / max((centre - source) * panel_inches, 0.5)
        ax.annotate('', xy=(centre, y + h + 0.01), xytext=(source, y + h + 0.01), arrowprops=dict(arrowstyle='-|>', color=GREY, lw=0.9, connectionstyle=f'arc3,rad={rad:.3f}', shrinkA=2, shrinkB=2))
    ax.text(0.5, y + h + 0.26, 'transferred model, never refitted', ha='center', va='bottom', fontsize=7.5, color=GREY)
    for centre in centres[len(train):]:
        ax.text(centre, y - 0.06, f"oracle: {cfg['n_folds']}-fold\ncross-fitted here", ha='center', va='top', fontsize=7, color=GREY)
    ax.text(0.5, 0.03, 'Gap = AUROC(oracle) − AUROC(transferred)      DiD = Gap(action-derived label) − Gap(physiology-derived label)', ha='center', va='bottom', fontsize=7.8)

def panel_gradient(ax, labels_cfg) -> None:
    blank(ax)
    panel_label(ax, 'b', 'labels, ordered by dependence on documented action')
    order = sorted(labels_cfg['labels'], key=lambda k: labels_cfg['labels'][k]['rank'])
    width, gap = (0.21, 0.053)
    xs = [0.02 + i * (width + gap) for i in range(len(order))]
    centres = [x + width / 2 for x in xs]
    y, h = (0.6, 0.2)
    for x, key in zip(xs, order, strict=True):
        colour = LABEL_COLOUR[key]
        box(ax, x, y, width, h, fc=colour, ec=colour)
        ax.text(x + width / 2, y + h / 2, LABEL_SHORT[key].replace(' ', '\n'), ha='center', va='center', fontsize=7.2, color='white', fontweight='bold', linespacing=1.1)
        ax.text(x + width / 2, y + h + 0.045, f"rank {labels_cfg['labels'][key]['rank']}", ha='center', va='bottom', fontsize=7, color=GREY)
    for contrast, level in zip(labels_cfg['primary_contrasts'], (0.44, 0.24), strict=True):
        a = centres[order.index(contrast['label_a'])]
        b = centres[order.index(contrast['label_b'])]
        colour = LABEL_COLOUR[contrast['label_a']]
        ax.plot([a, a, b, b], [y - 0.02, level, level, y - 0.02], color=colour, lw=0.9, clip_on=False)
        ax.text((a + b) / 2, level - 0.015, CONTRAST_DISPLAY[contrast['name']], ha='center', va='top', fontsize=7.4, color=colour)
    ax.annotate('', xy=(centres[-1], 0.08), xytext=(centres[0], 0.08), arrowprops=dict(arrowstyle='-|>', color=GREY, lw=0.8))
    ax.text((centres[0] + centres[-1]) / 2, 0.04, 'more of the label is determined by what a clinician recorded doing', ha='center', va='top', fontsize=7, color=GREY)

def panel_timeline(ax, labels_cfg, feats) -> None:
    task = labels_cfg['task']
    t_pred, horizon = (task['prediction_hour'], task['horizon_hours'])
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position(('axes', 0.12))
    panel_label(ax, 'c', 'one ICU stay')
    ax.axvspan(0, feats['window_hours'], color='#EDF2F7', lw=0, ymin=0.12, ymax=0.55)
    ax.axvspan(t_pred, horizon, color=BAND_GREY, lw=0, ymin=0.12, ymax=0.55)
    ax.axvline(t_pred, color=GREY, lw=1.0, ls=(0, (3, 2)), ymin=0.12, ymax=0.55)
    ax.text(0, 0.58, f"features [0, {feats['window_hours']:g}) h", ha='left', va='bottom', fontsize=7.2)
    ax.text((t_pred + horizon) / 2, 0.33, f'outcome window\n({t_pred:g}, {horizon:g}] h', ha='center', va='center', fontsize=7.2)
    ax.text(t_pred + horizon * 0.02, 0.74, f'prediction at hour {t_pred:g}', ha='left', va='bottom', fontsize=7.2, color=GREY)
    ax.set_xticks([0, t_pred, horizon])
    ax.set_xticklabels([f'{int(v)}' for v in (0, t_pred, horizon)])
    ax.set_xlabel('hours from ICU admission', labelpad=1)

def panel_hospitals(ax, manifest, eicu_cfg) -> None:
    blank(ax)
    panel_label(ax, 'd', 'hospital axis (eICU): the same protocol across sites')
    regions = sorted(manifest['prevalence_by_region'], key=lambda r: -r['n_hospitals'])
    total = sum((r['n_hospitals'] for r in regions))
    left, span = (0.02, 0.96)
    y, h = (0.42, 0.17)
    x = left
    for region in regions:
        w = span * region['n_hospitals'] / total
        box(ax, x, y, w, h, fc='#F2F2F2', ec=GREY, lw=0.7)
        ax.text(x + w / 2, y + h / 2, f"{region['n_hospitals']}", ha='center', va='center', fontsize=7.2)
        ax.text(x + w / 2, y + h + 0.04, region['region'], ha='center', va='bottom', fontsize=6.6, color=GREY)
        x += w
    hosp = manifest['hospitals']
    ax.text(left, y + h + 0.24, f"{hosp['n_hospitals']} hospitals → {hosp['n_included_hospitals']} after data-quality inclusion → {manifest['n_common_cohort']:,} stays in the common cohort", ha='left', va='bottom', fontsize=7.5)
    n_cv = eicu_cfg['designs']['hospital_cv']['n_folds']
    n_loro = len(eicu_cfg['designs']['leave_one_region_out']['regions'])
    ax.text(left, y - 0.09, f"pooled estimate: {n_cv} disjoint hospital groups, each tested once;   heterogeneity: leave-one-region-out, {n_loro} folds;   resampling unit: the {eicu_cfg['bootstrap_unit']}", ha='left', va='top', fontsize=7.2, color=GREY)

def fig01_main() -> None:
    use_style()
    recon = json.loads((fig01_RESULTS / 'recon' / 'final_recon.json').read_text('utf-8'))
    manifest = json.loads((fig01_RESULTS / 'eicu' / 'dataset_manifest.json').read_text('utf-8'))
    main_cfg = load_config('experiments/main.yaml')
    eicu_cfg = load_config('experiments/eicu.yaml')
    labels_cfg = load_config('labels/primary.yaml')
    feats = load_config('features/physiology_only.yaml')
    fig = plt.figure(figsize=(TEXT_WIDTH, 5.8))
    grid = GridSpec(3, 2, figure=fig, height_ratios=[1.15, 1.0, 0.95], width_ratios=[1.35, 1.0], hspace=0.3, wspace=0.16, left=0.015, right=0.985, top=0.97, bottom=0.05)
    panel_epochs(fig.add_subplot(grid[0, :]), recon, main_cfg)
    panel_gradient(fig.add_subplot(grid[1, 0]), labels_cfg)
    panel_timeline(fig.add_subplot(grid[1, 1]), labels_cfg, feats)
    panel_hospitals(fig.add_subplot(grid[2, :]), manifest, eicu_cfg)
    save(fig, 'fig01_design')

fig02_RESULTS = PROJECT_ROOT / 'results'

ACTION_COLOUR = {'frac_culture': OKABE_ITO['bluish_green'], 'frac_antibiotic': OKABE_ITO['reddish_purple'], 'frac_suspicion': OKABE_ITO['black']}

ACTION_DISPLAY = {'frac_culture': 'diagnostic culture sent', 'frac_antibiotic': 'antibiotic prescribed', 'frac_suspicion': 'suspicion of infection\n(both, in the required order)'}

def fig02_epoch_short(epoch: str) -> str:
    first, last = [part.strip() for part in epoch.split('-')]
    return f'{first}–{last[2:]}'

def shade_training(ax, epochs, train_epochs) -> None:
    last = max((i for i, e in enumerate(epochs) if e in train_epochs))
    ax.axvspan(-0.3, last + 0.5, color='#EDF2F7', lw=0, zorder=0)
    ax.annotate('training era', xy=(last + 0.42, 0.985), xycoords=('data', 'axes fraction'), ha='right', va='top', fontsize=6.6, color=GREY)

def fig02_main() -> None:
    use_style()
    recon = json.loads((fig02_RESULTS / 'recon' / 'final_recon.json').read_text('utf-8'))
    cfg = load_config('experiments/main.yaml')
    labels_cfg = load_config('labels/primary.yaml')
    prevalence = recon['prevalence_on_common_cohort']
    actions = recon['clinician_action_rates']
    kappa = recon['cohens_kappa']
    epochs = [row['epoch'] for row in prevalence]
    xs = list(range(len(epochs)))
    order = sorted(labels_cfg['labels'], key=lambda k: labels_cfg['labels'][k]['rank'])
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 3.4))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.89, bottom=0.42, wspace=0.42)
    ax = axes[0]
    for label in order:
        ax.plot(xs, [row[f'prev_{label}'] for row in prevalence], color=LABEL_COLOUR[label], marker='o', ms=3.2, lw=1.2, label=LABEL_DISPLAY[label])
    ax.set_ylabel('share of stays with the event')
    ax.set_title('(a)  label prevalence', loc='left')
    ax.set_ylim(0, 0.5)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.02, -0.3), fontsize=6.8, ncol=2, columnspacing=1.0, handletextpad=0.5)
    ax = axes[1]
    for key, colour in ACTION_COLOUR.items():
        ax.plot(xs, [row[key] for row in actions], color=colour, marker='o', ms=3.2, lw=1.2, label=ACTION_DISPLAY[key])
    ax.plot(xs, [row['prev_sepsis3'] for row in prevalence], color=LABEL_COLOUR['sepsis3'], marker='o', ms=3.2, lw=1.2, label='Sepsis-3 (the label)')
    ax.set_ylabel('share of stays')
    ax.set_title('(b)  what Sepsis-3 is made of', loc='left')
    ax.set_ylim(0, 0.7)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.02, -0.3), fontsize=6.8, ncol=1, handletextpad=0.5, labelspacing=0.35)
    ax = axes[2]
    for contrast in labels_cfg['primary_contrasts']:
        pair = next((k for k in (f"{contrast['label_a']}__{contrast['label_b']}", f"{contrast['label_b']}__{contrast['label_a']}") if k in kappa[0]))
        colour = LABEL_COLOUR[contrast['label_a']]
        ax.plot(xs, [row[pair] for row in kappa], color=colour, marker='o', ms=3.2, lw=1.2, label=f"{LABEL_SHORT[contrast['label_b']]} vs {LABEL_SHORT[contrast['label_a']]}")
    ax.set_ylabel("Cohen's $\\kappa$")
    ax.set_title('(c)  agreement within each contrast', loc='left')
    ax.set_ylim(0, 0.65)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.02, -0.3), fontsize=6.8, ncol=1, handletextpad=0.5, labelspacing=0.35)
    for ax in axes:
        shade_training(ax, epochs, set(cfg['train_epochs']))
        ax.set_xticks(xs)
        ax.set_xticklabels([fig02_epoch_short(e) for e in epochs], rotation=35, ha='right')
        ax.set_xlim(-0.3, len(epochs) - 0.7)
    save(fig, 'fig02_labels_by_epoch')

fig03_RESULTS = PROJECT_ROOT / 'results'

def fig03_epoch_short(epoch):
    parts = epoch.split('-')
    first = parts[0].strip()
    last = parts[1].strip()
    return first + '–' + last[2:]

def fig03_signed(x, places=4):
    return ('{:+.' + str(places) + 'f}').format(x).replace('-', '−')

def scalings_agreeing(estimate):
    block = estimate['sign_agreement_across_scalings']
    values = [block['gap'], block['gap_relative'], block['gap_logit']]
    sign = 1 if block['gap'] > 0 else -1
    agree = 0
    for v in values:
        if (1 if v > 0 else -1) == sign:
            agree = agree + 1
    return (agree, len(values))

def fig03_main():
    use_style()
    main_run = json.loads((fig03_RESULTS / 'main' / 'main_run.json').read_text('utf-8'))
    cfg = load_config('experiments/main.yaml')
    sesoi = cfg['sesoi']
    estimates = main_run['estimates']
    rows = []
    for i in range(len(cfg['contrasts'])):
        contrast = cfg['contrasts'][i]['name']
        action_label = CONTRAST_ACTION_LABEL[contrast]
        colour = LABEL_COLOUR[action_label]
        physio_label = cfg['contrasts'][i]['label_b']
        heading = CONTRAST_DISPLAY[contrast] + ' = Gap(' + LABEL_SHORT[action_label] + ') − Gap(' + LABEL_SHORT[physio_label] + ')'
        rows.append(row(heading, heading=True, rule_above=i > 0))
        for epoch in cfg['target_epochs']:
            for size in cfg['training_sizes']:
                est = estimates[contrast + '|' + epoch + '|' + size]
                agree, total = scalings_agreeing(est)
                verdict = VERDICT_SHORT[est['verdict']]
                notes = (fig03_signed(est['point']) + ' [' + fig03_signed(est['ci95'][0]) + ', ' + fig03_signed(est['ci95'][1]) + ']', '{:.4f}'.format(est['delta_min']), verdict + '  ' + str(agree) + '/' + str(total))
                rows.append(row(fig03_epoch_short(epoch) + '   ' + SIZE_DISPLAY[size], point=est['point'], lo=est['ci95'][0], hi=est['ci95'][1], colour=colour, marker=SIZE_MARKER[size], filled=SIZE_FILLED[size], starred=est['holm']['reject'], notes=notes))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 4.1))
    fig.subplots_adjust(left=0.17, right=0.45, top=0.93, bottom=0.18)
    reach = 0.0
    for r in rows:
        if r['point'] is not None:
            reach = max(reach, abs(r['lo']), abs(r['hi']))
    limit = max(reach * 1.12, sesoi * 1.6)
    draw(ax, rows, sesoi=sesoi, xlabel='difference in transfer gap (AUROC)', note_x=(1.1, 1.78, 2.02), note_headers=('DiD [95% CI]', '$\\delta_{\\min}$', 'verdict, scalings'), xlim=(-limit, limit))
    direction_key(ax, left='← action-derived label aged less', right='action-derived label aged more, the direction H1 predicts →')
    handles = []
    for size in cfg['training_sizes']:
        face = GREY if SIZE_FILLED[size] else 'white'
        handles.append(plt.Line2D([], [], marker=SIZE_MARKER[size], ls='none', ms=4.2, color=GREY, markerfacecolor=face, label=SIZE_DISPLAY[size] + ' training set'))
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=BAND_GREY, ec='none', label='practical equivalence, ±' + '{:.3f}'.format(sesoi) + ' AUROC'))
    family = 0
    for e in estimates.values():
        if 'holm' in e:
            family = family + 1
    handles.append(plt.Line2D([], [], marker='$\\bigstar$', ls='none', ms=6, color=GREY, label='rejects at Holm-corrected $\\alpha$ = ' + str(cfg['multiplicity']['alpha']) + ', family of ' + str(family)))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.47, 0.17), ncol=1, handletextpad=0.6)
    save(fig, 'fig03_main_forest')

fig04_RESULTS = PROJECT_ROOT / 'results'

def fig04_epoch_short(epoch: str) -> str:
    first, last = [part.strip() for part in epoch.split('-')]
    return f'{first}–{last[2:]}'

def fig04_main() -> None:
    use_style()
    run = json.loads((fig04_RESULTS / 'main' / 'main_run.json').read_text('utf-8'))
    cfg = load_config('experiments/main.yaml')
    labels_cfg = load_config('labels/primary.yaml')
    epochs = cfg['target_epochs']
    sizes = cfg['training_sizes']
    order = sorted(labels_cfg['labels'], key=lambda k: labels_cfg['labels'][k]['rank'])
    cells: dict[tuple, list[float]] = defaultdict(list)
    for row in run['gaps']:
        key = (row['training_size'], row['test_epoch'], row['label'], row['model'])
        cells[key].append(row['gap'])
    fig, axes = plt.subplots(1, len(sizes), figsize=(TEXT_WIDTH, 2.9), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.79, top=0.88, bottom=0.17, wspace=0.08)
    xs = list(range(len(epochs)))
    for ax, size in zip(axes, sizes, strict=True):
        ax.axhline(0.0, color=GREY, lw=0.7, zorder=1)
        for label in order:
            colour = LABEL_COLOUR[label]
            per_model = [[st.mean(cells[size, epoch, label, model]) for model in cfg['models']] for epoch in epochs]
            means = [st.mean(vals) for vals in per_model]
            ax.fill_between(xs, [min(v) for v in per_model], [max(v) for v in per_model], color=colour, alpha=0.13, lw=0, zorder=2)
            ax.plot(xs, means, color=colour, marker='o', ms=3.4, lw=1.2, zorder=3, label=LABEL_DISPLAY[label])
        ax.set_xticks(xs)
        ax.set_xticklabels([fig04_epoch_short(e) for e in epochs])
        ax.set_xlim(-0.25, len(epochs) - 0.75)
        ax.set_title(f'{SIZE_DISPLAY[size]} training set')
        ax.set_xlabel('target epoch')
    axes[0].set_ylabel('transfer gap (AUROC)')
    axes[0].annotate('above 0: the transferred model\nis worse — the label aged', xy=(0.03, 0.96), xycoords='axes fraction', va='top', ha='left', fontsize=6.8, color=GREY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.8, 0.86), title='label', alignment='left')
    fig.legends[0].get_title().set_fontsize(7.5)
    fig.text(0.8, 0.34, 'line: mean over the three\nmodel families and all seeds\nband: range over the families', fontsize=6.8, color=GREY, va='top')
    save(fig, 'fig04_gap_by_label')

fig05_RESULTS = PROJECT_ROOT / 'results'

MODEL_DISPLAY = {'logreg': 'logistic regression', 'xgboost': 'XGBoost', 'gru': 'GRU'}

def fig05_epoch_short(epoch: str) -> str:
    first, last = [part.strip() for part in epoch.split('-')]
    return f'{first}–{last[2:]}'

def fig05_main() -> None:
    use_style()
    curve = json.loads((fig05_RESULTS / 'main' / 'learning_curve.json').read_text('utf-8'))
    cfg = load_config('experiments/main.yaml')
    labels_cfg = load_config('labels/primary.yaml')
    order = sorted(labels_cfg['labels'], key=lambda k: labels_cfg['labels'][k]['rank'])
    highlight = cfg['target_epochs'][-1]
    series: dict[tuple, list[dict]] = defaultdict(list)
    for row in curve['rows']:
        series[row['model'], row['label'], row['epoch']].append(row)
    models = cfg['models']
    fig, axes = plt.subplots(1, len(models), figsize=(TEXT_WIDTH, 2.9), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.8, top=0.88, bottom=0.18, wspace=0.08)
    for ax, model in zip(axes, models, strict=True):
        for label in order:
            colour = LABEL_COLOUR[label]
            for epoch in cfg['target_epochs']:
                rows = sorted(series[model, label, epoch], key=lambda r: r['n_positives'])
                ax.plot([r['n_positives'] for r in rows], [r['auroc_mean'] for r in rows], color=colour, marker='o', ms=2.6, lw=1.0, alpha=0.85, zorder=2, label=LABEL_DISPLAY[label] if epoch == cfg['target_epochs'][0] else None)
        point = [r for r in series[model, 'sepsis3', highlight] if r['fraction'] == 1.0][0]
        ax.plot([point['n_positives']], [point['auroc_mean']], marker='o', ms=7.5, mfc='none', mec=OKABE_ITO['reddish_purple'], mew=1.2, zorder=4)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(ticker.FixedLocator([250, 500, 1000, 2000, 4000]))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_title(MODEL_DISPLAY[model])
        ax.set_xlabel('positives in the oracle training set')
    axes[0].set_ylabel('oracle AUROC')
    point = [r for r in series[models[0], 'sepsis3', highlight] if r['fraction'] == 1.0][0]
    axes[0].annotate(f"Sepsis-3 in {fig05_epoch_short(highlight)}:\n{point['n_positives']:,} positives,\nthe smallest oracle\nof any cell", xy=(point['n_positives'], point['auroc_mean']), xytext=(0.02, 0.99), textcoords='axes fraction', ha='left', va='top', fontsize=6.8, color=OKABE_ITO['reddish_purple'], arrowprops=dict(arrowstyle='-', lw=0.7, color=OKABE_ITO['reddish_purple'], shrinkA=1, shrinkB=6))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.815, 0.86), title='label', alignment='left')
    fig.legends[0].get_title().set_fontsize(7.5)
    fig.text(0.815, 0.42, f"one line per label per\ntarget epoch, at 25, 50,\n75 and 100% of that\noracle's training set\n({curve['n_seeds']} seeds each)", fontsize=6.8, color=GREY, va='top')
    save(fig, 'fig05_learning_curve')

fig06_RESULTS = PROJECT_ROOT / 'results'

def fig06_signed(x, places=4):
    return ('{:+.' + str(places) + 'f}').format(x).replace('-', '−')

def thousands(n):
    return '{:,}'.format(n)

def fig06_main():
    use_style()
    run = json.loads((fig06_RESULTS / 'eicu' / 'eicu_run.json').read_text('utf-8'))
    cfg = load_config('experiments/eicu.yaml')
    sesoi = cfg['sesoi']
    sizes = list(cfg['training_sizes']) + list(cfg['training_sizes_secondary'])
    secondary_sizes = set(cfg['training_sizes_secondary'])
    estimates = run['estimates']
    primary = {}
    for k in estimates:
        e = estimates[k]
        if not e['secondary'] and e['contrast'] == 'DiD_aki':
            primary[k] = e
    pooled = {}
    by_region = {}
    for k in primary:
        e = primary[k]
        if e['design'] == 'hospital_cv':
            pooled[e['training_size']] = e
        elif e['design'] == 'loro':
            if e['split'] not in by_region:
                by_region[e['split']] = {}
            by_region[e['split']][e['training_size']] = e
    order = sorted(by_region, key=lambda r: -by_region[r]['full']['point'])
    colour = LABEL_COLOUR[CONTRAST_ACTION_LABEL['DiD_aki']]

    def estimate_row(est, size):
        is_secondary = size in secondary_sizes
        text = SIZE_DISPLAY[size]
        if is_secondary:
            text = text + '  (secondary)'
        if 'holm' in est:
            holm = '{:.4f}'.format(est['holm']['p_holm'])
        else:
            holm = '—'
        notes = (fig06_signed(est['point']) + ' [' + fig06_signed(est['ci95'][0]) + ', ' + fig06_signed(est['ci95'][1]) + ']', '{:.4f}'.format(est['delta_min']), holm, VERDICT_SHORT[est['verdict']])
        return row(text, point=est['point'], lo=est['ci95'][0], hi=est['ci95'][1], colour=colour, marker=SIZE_MARKER[size], filled=SIZE_FILLED[size], starred=est.get('holm', {}).get('reject', False), notes=notes, muted=is_secondary)
    n_pooled_hospitals = sum(pooled['full']['fold_hospitals'])
    head = 'pooled over ' + str(pooled['full']['n_folds_pooled']) + ' disjoint hospital groups — ' + str(n_pooled_hospitals) + ' hospitals, ' + thousands(pooled['full']['n_test']) + ' stays'
    rows = [row(head, heading=True)]
    for size in sizes:
        if size in pooled:
            rows.append(estimate_row(pooled[size], size))
    for i in range(len(order)):
        region = order[i]
        first = by_region[region]['full']
        text = region + ' — ' + str(first['n_test_hospitals']) + ' hospitals, ' + thousands(first['n_test']) + ' stays'
        rows.append(row(text, heading=True, rule_above=i == 0))
        for size in sizes:
            if size in by_region[region]:
                rows.append(estimate_row(by_region[region][size], size))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 5.9))
    fig.subplots_adjust(left=0.16, right=0.45, top=0.95, bottom=0.12)
    reach = 0.0
    for r in rows:
        if r['point'] is not None:
            reach = max(reach, abs(r['lo']), abs(r['hi']))
    limit = max(reach * 1.1, sesoi * 1.6)
    draw(ax, rows, sesoi=sesoi, xlabel='difference in transfer gap (AUROC)', note_x=(1.1, 1.8, 2.04, 2.23), note_headers=('DiD [95% CI]', '$\\delta_{\\min}$', 'Holm $p$', 'verdict'), xlim=(-limit, limit), heading_x=-0.52)
    ax.axvline(pooled['full']['point'], color=GREY, lw=0.8, ls=(0, (4, 2)), zorder=2)
    direction_key(ax, left='← urine-AKI, the action-dependent label, aged less', right='urine-AKI aged more →', y=-0.085, line_gap=0.028)
    handles = []
    for size in sizes:
        face = GREY if SIZE_FILLED[size] else 'white'
        label = SIZE_DISPLAY[size] + ' training set'
        if size in secondary_sizes:
            label = label + ' — secondary'
        handles.append(plt.Line2D([], [], marker=SIZE_MARKER[size], ls='none', ms=4.2, color=GREY, markerfacecolor=face, label=label))
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=BAND_GREY, ec='none', label='practical equivalence, ±' + '{:.3f}'.format(sesoi) + ' AUROC'))
    family = 0
    for e in estimates.values():
        if 'holm' in e:
            family = family + 1
    handles.append(plt.Line2D([], [], marker='$\\bigstar$', ls='none', ms=6, color=GREY, label='rejects at Holm-corrected $\\alpha$ = ' + str(cfg['multiplicity']['alpha']) + ', family of ' + str(family)))
    handles.append(plt.Line2D([], [], ls=(0, (4, 2)), lw=0.8, color=GREY, label='pooled estimate, full training set'))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.47, 0.115), ncol=1, handletextpad=0.6)
    save(fig, 'fig06_eicu_regions')

fig07_RESULTS = PROJECT_ROOT / 'results'

SHOWN = ['aki_creatinine', 'aki_urine']

def fig07_main() -> None:
    use_style()
    manifest = json.loads((fig07_RESULTS / 'eicu' / 'dataset_manifest.json').read_text('utf-8'))
    cultures = json.loads((fig07_RESULTS / 'eicu' / 'culture_by_hospital.json').read_text('utf-8'))
    deff = json.loads((fig07_RESULTS / 'eicu' / 'design_effect.json').read_text('utf-8'))
    cfg = load_config('experiments/eicu.yaml')
    region_of = {h['hospitalid']: h['region'] or 'Unknown' for h in cultures['per_hospital']}
    by_region: dict[str, list[dict]] = {}
    for hospital in manifest['prevalence_by_hospital']:
        by_region.setdefault(region_of[hospital['hospitalid']], []).append(hospital)
    means = {row['region']: row for row in manifest['prevalence_by_region']}
    order = sorted(by_region, key=lambda r: -means[r]['aki_urine'])
    rng = np.random.default_rng(cfg['bootstrap']['seed'])
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.9))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.17)
    offsets = np.linspace(-0.19, 0.19, len(SHOWN))
    for i, region in enumerate(order):
        hospitals = by_region[region]
        for label, offset in zip(SHOWN, offsets, strict=True):
            colour = LABEL_COLOUR[label]
            values = [h[label] for h in hospitals]
            jitter = rng.uniform(-0.055, 0.055, len(values))
            ax.scatter(np.full(len(values), i + offset) + jitter, values, s=7, color=colour, alpha=0.55, lw=0, zorder=2, label=LABEL_DISPLAY[label] if i == 0 else None)
            ax.plot([i + offset - 0.1, i + offset + 0.1], [means[region][label]] * 2, color=colour, lw=1.8, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{r}\n{means[r]['n_hospitals']} hospitals, {means[r]['n_stays']:,} stays" for r in order])
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("share of a hospital's stays with the event")
    icc = {row['split']: row['labels'] for row in deff['per_split']}['cv (all)']
    note = '   '.join((f"{LABEL_SHORT[label]}: ICC {icc[label]['icc']:.3f}, DEFF {icc[label]['deff']:.1f}" for label in SHOWN))
    ax.annotate(f"between-hospital clustering over all {manifest['hospitals']['n_included_hospitals']} hospitals —   " + note, xy=(0.5, 0.985), xycoords='axes fraction', ha='center', va='top', fontsize=7, color=GREY)
    ax.annotate('bar: region mean;  dot: one hospital', xy=(0.995, 0.86), xycoords='axes fraction', ha='right', va='top', fontsize=6.8, color=GREY)
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.84), fontsize=7.2)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    save(fig, 'fig07_eicu_urine_by_region')

tables_RESULTS = PROJECT_ROOT / 'results'

CSV_DIR = tables_RESULTS / 'tables'

EPOCH_KEY = {'2008 - 2010': 'EarlyA', '2011 - 2013': 'EarlyB', '2014 - 2016': 'MidA', '2017 - 2019': 'MidB', '2020 - 2022': 'Late'}

LABEL_KEY = {'aki_creatinine': 'AkiCr', 'sofa_dysfunction': 'Sofa', 'aki_urine': 'AkiUo', 'sepsis3': 'Sepsis'}

LABEL_NAME = {'aki_creatinine': 'AKI (creatinine)', 'sofa_dysfunction': 'SOFA deterioration', 'aki_urine': 'AKI (urine output)', 'sepsis3': 'Sepsis-3'}

SIZE_KEY = {'full': 'Full', 'matched': 'Matched', 'event_matched': 'EventMatched'}

SIZE_NAME = {'full': 'full', 'matched': 'matched', 'event_matched': 'event-matched'}

RULE = '\\midrule'

def Cell(csv_value, tex=None):
    return {'csv': csv_value, 'tex': str(csv_value) if tex is None else tex}

def macro(name: str, value) -> Cell:
    return Cell(value, f'$\\{name}$')

def word_macro(name: str, value) -> Cell:
    return Cell(value, f'\\{name}')

def text(value: str) -> Cell:
    return Cell(value, value)

def tables_signed(x: float) -> str:
    return f'{x:+.4f}'

def places(x: float, n: int=3) -> str:
    return f'{x:.{n}f}'

def write(stem: str, caption: str, colspec: str, header: list[tuple[str, str]], rows: list, notes: list[str], font: str='\\small') -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f'{stem}.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([h[0] for h in header])
        for row in rows:
            if row is RULE or isinstance(row, str):
                continue
            writer.writerow([c['csv'] for c in row])
    lines = ['% LaTeX layout of the table, kept for reference only.', '% The CSV written above is the output of this step.', '\\begin{widetable}[t]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{tab:{stem}}}', font, f'\\begin{{tabular}}{{{colspec}}}', '\\toprule', ' & '.join((h[1] for h in header)) + ' \\\\', '\\midrule']
    for row in rows:
        if row is RULE:
            lines.append('\\midrule')
        elif isinstance(row, str):
            lines.append(f'\\multicolumn{{{len(header)}}}{{l}}{{\\emph{{{row}}}}} \\\\')
        else:
            lines.append(' & '.join((c['tex'] for c in row)) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    if notes:
        lines.append('\\vspace{2pt}')
        lines.append('\\begin{minipage}{\\linewidth}\\footnotesize')
        lines += [note + '\\\\' for note in notes]
        lines.append('\\end{minipage}')
    lines.append('\\end{widetable}')
    n_rows = sum((1 for r in rows if not isinstance(r, str) and r is not RULE))
    print(f'  {stem}: {n_rows} rows -> {csv_path.name}')

def table_cohort(recon, coverage) -> None:
    rows = []
    prevalence = {r['epoch']: r for r in recon['prevalence_on_common_cohort']}
    actions = {r['epoch']: r for r in recon['clinician_action_rates']}
    for row in recon['common_cohort']:
        epoch, key = (row['epoch'], EPOCH_KEY[row['epoch']])
        cells = [word_macro(f'Epoch{key}', epoch), macro(f'NCohort{key}', row['n_stays']), macro(f'NCommon{key}', row['n_common_cohort'])]
        cells += [macro(f'Prev{LABEL_KEY[label]}{key}', places(prevalence[epoch][f'prev_{label}'])) for label in LABEL_KEY]
        cells += [macro(f'FracCulture{key}', places(actions[epoch]['frac_culture'])), macro(f'FracAntibiotic{key}', places(actions[epoch]['frac_antibiotic']))]
        rows.append(cells)
    rows.append(RULE)
    total = [text('all epochs'), macro('CohortTotal', coverage['cohort']['total']), macro('CommonCohortTotal', sum((r['n_common_cohort'] for r in recon['common_cohort'])))]
    n_common = sum((r['n'] for r in recon['prevalence_on_common_cohort']))
    n_all = sum((r['n'] for r in recon['clinician_action_rates']))
    total += [macro(f'Prev{LABEL_KEY[label]}All', places(sum((r[f'n_{label}'] for r in recon['prevalence_on_common_cohort'])) / n_common)) for label in LABEL_KEY]
    total += [macro(f'Frac{name}All', places(sum((r[field] * r['n'] for r in recon['clinician_action_rates'])) / n_all)) for name, field in (('Culture', 'frac_culture'), ('Antibiotic', 'frac_antibiotic'))]
    rows.append(total)
    write('cohort', 'Cohort and label prevalence by epoch, MIMIC-IV.', 'lrrrrrrrr', [('Epoch', 'Epoch'), ('ICU stays', 'ICU stays'), ('Common cohort', 'Common cohort'), ('AKI (cr)', 'AKI (cr)'), ('SOFA', 'SOFA'), ('AKI (UO)', 'AKI (UO)'), ('Sepsis-3', 'Sepsis-3'), ('Culture', 'Culture'), ('Antibiotic', 'Antibiotic')], rows, ['Prevalence is on the common cohort: the stays on which all four labels are evaluable. The culture and antibiotic columns are over every stay of the cohort in the epoch, since an action is recorded or not regardless of which labels can be computed.'], font='\\footnotesize')

def table_labels(labels_cfg, manifest, recon) -> None:
    rank_macro = {'aki_creatinine': 'RankAkiCr', 'sofa_dysfunction': 'RankSofa', 'aki_urine': 'RankAkiUo', 'sepsis3': 'RankSepsis'}
    availability = {'aki_creatinine': 'both datasets, no restriction', 'sofa_dysfunction': 'both datasets, no restriction', 'aki_urine': 'both datasets; eICU charting varies sharply by hospital', 'sepsis3': None}
    n_common = sum((r['n'] for r in recon['prevalence_on_common_cohort']))
    rows = []
    for label, key in LABEL_KEY.items():
        note = availability[label]
        overall = places(sum((r[f'n_{label}'] for r in recon['prevalence_on_common_cohort'])) / n_common)
        note_cell = text(note) if note is not None else Cell(f"eICU: {manifest['hospitals']['n_sepsis_hospitals']} culture-charting hospitals only", 'eICU: $\\EicuSepsisHospitals$ culture-charting hospitals only')
        rows.append([text(LABEL_NAME[label]), macro(rank_macro[label], labels_cfg['labels'][label]['rank']), word_macro(f'Crit{key}', labels_cfg['labels'][label]['criterion']), macro(f'Prev{key}All', overall), macro(f'EicuPrev{key}', places(manifest['prevalence_overall'][label])), note_cell])
    write('labels', 'The four primary labels: definition, action dependence, and availability in each dataset.', 'llp{0.24\\linewidth}rrp{0.15\\linewidth}', [('Label', 'Label'), ('Rank', 'Rank'), ('Criterion', 'Criterion'), ('MIMIC-IV prevalence', 'MIMIC-IV'), ('eICU prevalence', 'eICU'), ('Availability', 'Availability')], rows, ['Rank is the registered action-dependence order: rank $\\RankAkiCr$ depends on a laboratory assay alone, rank $\\RankSepsis$ on documented clinician actions. Prevalence columns are shares of the common cohort of each dataset.', 'SOFA deterioration and Sepsis-3 are rise-based and have no self-consistent prevalent case, so no prevalent-case exclusion is applied. The eICU prevalence of Sepsis-3 is over all included hospitals, most of which record few cultures; the Sepsis-3 contrast uses only the culture-charting hospitals.'], font='\\footnotesize')

def table_main(run, cfg) -> None:
    rows = []
    for i, contrast in enumerate((c['name'] for c in cfg['contrasts'])):
        short = 'Sepsis' if contrast == 'DiD_sepsis' else 'Aki'
        if i:
            rows.append(RULE)
        rows.append('DiD$_{\\mathrm{sepsis}}$ = Gap(Sepsis-3) $-$ Gap(SOFA deterioration)' if contrast == 'DiD_sepsis' else 'DiD$_{\\mathrm{AKI}}$ = Gap(urine-AKI) $-$ Gap(creatinine-AKI)')
        for epoch in cfg['target_epochs']:
            for size in cfg['training_sizes']:
                e = run['estimates'][f'{contrast}|{epoch}|{size}']
                stem = f'Did{short}{EPOCH_KEY[epoch]}{SIZE_KEY[size]}'
                rows.append([word_macro(f'Epoch{EPOCH_KEY[epoch]}', epoch), text(SIZE_NAME[size]), macro(stem, tables_signed(e['point'])), Cell(f"[{tables_signed(e['ci95'][0])}, {tables_signed(e['ci95'][1])}]", f'$[\\{stem}Lo,\\ \\{stem}Hi]$'), macro(f'{stem}DeltaMin', f"{e['delta_min']:.4f}"), macro(f'{stem}PHolm', f"{e['holm']['p_holm']:.4f}"), word_macro(f'{stem}Verdict', e['verdict'])])
    write('main_results', 'Primary MIMIC-IV estimates. Twelve cells, one family, Holm-corrected.', 'llrrrrp{0.17\\linewidth}', [('Epoch', 'Epoch'), ('Training set', 'Training set'), ('DiD', 'DiD'), ('95% CI', '\\CILevel\\% CI'), ('delta_min', '$\\delta_{\\min}$'), ('Holm p', 'Holm $p$'), ('Verdict', 'Verdict')], rows, ['$\\delta_{\\min}$ is the smallest equivalence margin the data support at the \\TOSTLevel\\% level; a verdict of practically equivalent requires it to fall below the SESOI of \\SESOI.', 'Both training-size variants are primary. The family for the Holm correction is all \\FamilySize\\ cells. Verdicts apply the decision rules to the unadjusted intervals; the Holm column shows which cells also survive the multiplicity correction.'])

def table_eicu(run, manifest, cfg) -> None:
    estimates = run['estimates']
    primary_sizes = cfg['training_sizes']
    secondary_sizes = cfg['training_sizes_secondary']
    regions = [e['split'] for e in estimates.values() if e['design'] == 'loro' and e['training_size'] == 'full' and (not e['secondary'])]
    regions.sort(key=lambda r: -estimates[f'DiD_aki|loro:{r}|full']['point'])
    by_region = {row['region']: row for row in manifest['prevalence_by_region']}

    def block(key_of, where_name, sizes, contrast_short):
        out = []
        for size in sizes:
            key = key_of(size)
            e = estimates[key]
            stem = f"Eicu{('Sec' if e['secondary'] else '')}Did{contrast_short}{('Cv' if e['design'] == 'hospital_cv' else e['split'])}{SIZE_KEY[size]}"
            out.append([where_name[0] if size == sizes[0] else text(''), where_name[1] if size == sizes[0] else text(''), text(SIZE_NAME[size]), macro(stem, tables_signed(e['point'])), Cell(f"[{tables_signed(e['ci95'][0])}, {tables_signed(e['ci95'][1])}]", f'$[\\{stem}Lo,\\ \\{stem}Hi]$'), macro(f'{stem}DeltaMin', f"{e['delta_min']:.4f}"), macro(f'{stem}PHolm', f"{e['holm']['p_holm']:.4f}") if 'holm' in e else text('---'), word_macro(f'{stem}Verdict', e['verdict'])])
        return out
    pooled_where = (text('pooled'), Cell(manifest['hospitals']['n_included_hospitals'], '\\EicuHospitalsIncluded'))
    rows: list = ['Primary: DiD$_{\\mathrm{AKI}}$ along the hospital axis']
    rows += block(lambda s: f'DiD_aki|cv|{s}', pooled_where, primary_sizes, 'Aki')
    for region in regions:
        where = (text(region), macro(f"EicuNHosp{region.replace(' ', '')}", by_region[region]['n_hospitals']))
        rows += block(lambda s, r=region: f'DiD_aki|loro:{r}|{s}', where, primary_sizes, 'Aki')
    rows.append(RULE)
    rows.append('Secondary: the same contrast with positives matched as well as patients')
    rows += block(lambda s: f'DiD_aki|cv|{s}', pooled_where, secondary_sizes, 'Aki')
    for region in regions:
        where = (text(region), macro(f"EicuNHosp{region.replace(' ', '')}", by_region[region]['n_hospitals']))
        rows += block(lambda s, r=region: f'DiD_aki|loro:{r}|{s}', where, secondary_sizes, 'Aki')
    rows.append(RULE)
    rows.append('Secondary: DiD$_{\\mathrm{sepsis}}$ on the culture-charting subset')
    sepsis_where = (text('pooled'), Cell(manifest['hospitals']['n_sepsis_hospitals'], '\\EicuSepsisHospitals'))
    rows += block(lambda s: f'SECONDARY|DiD_sepsis|cv|{s}', sepsis_where, primary_sizes, 'Sepsis')
    write('eicu', 'eICU replication along the hospital axis. The pooled estimate and the regional folds come from two designs and are read together.', 'lrlrlrrp{0.13\\linewidth}', [('Split', 'Split'), ('Hospitals', 'Hospitals'), ('Training set', 'Training set'), ('DiD', 'DiD'), ('95% CI', '\\CILevel\\% CI'), ('delta_min', '$\\delta_{\\min}$'), ('Holm p', 'Holm $p$'), ('Verdict', 'Verdict')], rows, ['The resampling unit is the hospital. The pooled row averages \\EicuNFoldsPooled\\ disjoint hospital groups covering every included hospital; it measures transfer between hospitals of the same mix, while each regional row measures transfer into one held-out region, so the pooled row is not an average of the regional rows.', 'The Holm correction as run covered \\EicuHolmFamily\\ estimates rather than the \\FamilySize\\ the design declares (see the preregistration, Section 13). A meaningful drift asymmetry is in the predicted direction, urine-output AKI ageing more.', 'The Sepsis-3 subset was declared underpowered for equivalence before it was run. Verdicts apply the decision rules to the unadjusted intervals; the Holm column shows which of them also survive the multiplicity correction.'], font='\\footnotesize')

def tables_main() -> None:
    recon = json.loads((tables_RESULTS / 'recon' / 'final_recon.json').read_text('utf-8'))
    coverage = json.loads((tables_RESULTS / 'recon' / 'concept_coverage.json').read_text('utf-8'))
    main_run = json.loads((tables_RESULTS / 'main' / 'main_run.json').read_text('utf-8'))
    eicu_run = json.loads((tables_RESULTS / 'eicu' / 'eicu_run.json').read_text('utf-8'))
    manifest = json.loads((tables_RESULTS / 'eicu' / 'dataset_manifest.json').read_text('utf-8'))
    table_cohort(recon, coverage)
    table_labels(load_config('labels/primary.yaml'), manifest, recon)
    table_main(main_run, load_config('experiments/main.yaml'))
    table_eicu(eicu_run, manifest, load_config('experiments/eicu.yaml'))

STAGES = {
    'fig01': fig01_main,
    'fig02': fig02_main,
    'fig03': fig03_main,
    'fig04': fig04_main,
    'fig05': fig05_main,
    'fig06': fig06_main,
    'fig07': fig07_main,
    'tables': tables_main,
}

if __name__ == '__main__':
    argv = sys.argv[1:]
    chosen = [argv[0]] if argv and argv[0] in STAGES else list(STAGES)
    rest = argv[1:] if argv and argv[0] in STAGES else argv
    for name in chosen:
        sys.argv = [name, *rest]
        print('== ' + name + ' ==')
        STAGES[name]()
