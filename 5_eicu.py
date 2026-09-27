from __future__ import annotations
import copy
import sys
import yaml
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import argparse
import duckdb
import hashlib
import json
import numpy as np
import os
import polars as pl
import time
import warnings

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


MIN_LOS_MINUTES = 12 * 60

MIN_AGE = 18

MIN_STAYS_PER_HOSPITAL = 100

MIN_CREATININE_COVERAGE = 0.5

HORIZON_MINUTES = 48 * 60

SEPSIS_MIN_CULTURE_RATE = 0.2

def register(con: duckdb.DuckDBPyConnection, tables: list[str]) -> None:
    root = parquet_dir('eicu')
    for t in tables:
        path = root / f'{t}.parquet'
        if not path.exists():
            raise FileNotFoundError(path)
        con.execute(f"CREATE OR REPLACE VIEW e_{t.lower()} AS SELECT * FROM read_parquet('{path.as_posix()}')")

def build_cohort(con: duckdb.DuckDBPyConnection) -> None:
    register(con, ['patient', 'hospital'])
    con.execute(f"\n        CREATE OR REPLACE TABLE ecohort AS\n        WITH ranked AS (\n            SELECT p.patientunitstayid, p.uniquepid, p.hospitalid,\n                   p.unitdischargeoffset, p.gender, p.unittype,\n                   p.unitadmitsource, p.hospitaldischargeyear,\n                   p.apacheadmissiondx,\n                   CASE WHEN p.age = '> 89' THEN 90\n                        ELSE TRY_CAST(p.age AS INTEGER) END AS age,\n                   TRY_CAST(p.admissionweight AS DOUBLE) AS weight_kg,\n                   ROW_NUMBER() OVER (\n                       PARTITION BY p.uniquepid\n                       ORDER BY p.hospitaldischargeyear,\n                                p.patienthealthsystemstayid, p.unitvisitnumber\n                   ) AS stay_rank\n            FROM e_patient p\n        )\n        SELECT r.* EXCLUDE (stay_rank),\n               h.region, h.numbedscategory, h.teachingstatus\n        FROM ranked r\n        LEFT JOIN e_hospital h USING (hospitalid)\n        WHERE r.stay_rank = 1\n          AND r.unitdischargeoffset >= {MIN_LOS_MINUTES}\n          AND r.age >= {MIN_AGE}\n        ")

def hospital_quality(con: duckdb.DuckDBPyConnection) -> None:
    register(con, ['lab', 'treatment'])
    con.execute(f"\n        CREATE OR REPLACE TABLE ehospital AS\n        WITH creat AS (\n            SELECT DISTINCT patientunitstayid FROM e_lab\n            WHERE lower(labname) LIKE 'creatinine%' AND labresult IS NOT NULL\n              AND labresultoffset BETWEEN 0 AND {HORIZON_MINUTES}\n        ), cult AS (\n            SELECT DISTINCT patientunitstayid FROM e_treatment\n            WHERE lower(treatmentstring) LIKE '%cultures%'\n        )\n        SELECT c.hospitalid,\n               any_value(c.region) AS region,\n               any_value(c.numbedscategory) AS beds,\n               any_value(c.teachingstatus) AS teaching,\n               count(*) AS n_stays,\n               avg(CASE WHEN cr.patientunitstayid IS NOT NULL THEN 1.0 ELSE 0 END)\n                   AS frac_creatinine,\n               avg(CASE WHEN cu.patientunitstayid IS NOT NULL THEN 1.0 ELSE 0 END)\n                   AS frac_culture\n        FROM ecohort c\n        LEFT JOIN creat cr ON cr.patientunitstayid = c.patientunitstayid\n        LEFT JOIN cult  cu ON cu.patientunitstayid = c.patientunitstayid\n        GROUP BY c.hospitalid\n        ")
    con.execute(f'\n        ALTER TABLE ehospital ADD COLUMN included BOOLEAN;\n        UPDATE ehospital SET included =\n            (n_stays >= {MIN_STAYS_PER_HOSPITAL}\n             AND frac_creatinine >= {MIN_CREATININE_COVERAGE});\n        ALTER TABLE ehospital ADD COLUMN sepsis_subset BOOLEAN;\n        UPDATE ehospital SET sepsis_subset =\n            (included AND frac_culture >= {SEPSIS_MIN_CULTURE_RATE});\n        ')

def summary(con: duckdb.DuckDBPyConnection) -> dict:
    row = con.execute('\n        SELECT count(*) AS n_hospitals,\n               sum(n_stays) AS n_stays,\n               sum(CASE WHEN included THEN 1 ELSE 0 END) AS n_included_hospitals,\n               sum(CASE WHEN included THEN n_stays ELSE 0 END) AS n_included_stays,\n               sum(CASE WHEN sepsis_subset THEN 1 ELSE 0 END) AS n_sepsis_hospitals,\n               sum(CASE WHEN sepsis_subset THEN n_stays ELSE 0 END) AS n_sepsis_stays\n        FROM ehospital\n        ').fetchone()
    keys = ['n_hospitals', 'n_stays', 'n_included_hospitals', 'n_included_stays', 'n_sepsis_hospitals', 'n_sepsis_stays']
    return dict(zip(keys, [int(v) for v in row], strict=True))

MINUTES = 60.0

def _cfg() -> dict:
    return load_config('concepts/eicu.yaml')

def lab(con: duckdb.DuckDBPyConnection, concept: str, max_hours: float=96.0, lookback_hours: float=0.0) -> pl.DataFrame:
    spec = _cfg()['labs'][concept]
    like = ' OR '.join((f"lower(l.labname) LIKE '{n}%'" for n in spec['names']))
    lo, hi = spec['valid']
    return con.execute(f'\n        SELECT c.patientunitstayid AS stay_id,\n               l.labresultoffset / {MINUTES} AS hours,\n               l.labresult AS value\n        FROM e_lab l\n        JOIN ecohort c USING (patientunitstayid)\n        WHERE ({like}) AND l.labresult BETWEEN {lo} AND {hi}\n          AND l.labresultoffset BETWEEN {-int(lookback_hours * 60)}\n                                    AND {int(max_hours * 60)}\n        ').pl()

def vital(con: duckdb.DuckDBPyConnection, concept: str, max_hours: float=96.0) -> pl.DataFrame:
    cfg = _cfg()
    parts = []
    spec = cfg['vitals'].get(concept)
    if spec:
        lo, hi = spec['valid']
        parts.append(f"SELECT c.patientunitstayid AS stay_id,\n                       v.observationoffset / {MINUTES} AS hours,\n                       v.{spec['column']} AS value\n                FROM e_vitalperiodic v JOIN ecohort c USING (patientunitstayid)\n                WHERE v.{spec['column']} BETWEEN {lo} AND {hi}\n                  AND v.observationoffset BETWEEN 0 AND {int(max_hours * 60)}")
    spec_a = cfg['vitals_aperiodic'].get(concept)
    if spec_a:
        lo, hi = spec_a['valid']
        parts.append(f"SELECT c.patientunitstayid AS stay_id,\n                       a.observationoffset / {MINUTES} AS hours,\n                       a.{spec_a['column']} AS value\n                FROM e_vitalaperiodic a JOIN ecohort c USING (patientunitstayid)\n                WHERE a.{spec_a['column']} BETWEEN {lo} AND {hi}\n                  AND a.observationoffset BETWEEN 0 AND {int(max_hours * 60)}")
    if not parts:
        raise KeyError(concept)
    return con.execute(' UNION ALL '.join(parts)).pl()

def gcs_total(con: duckdb.DuckDBPyConnection, max_hours: float=96.0) -> pl.DataFrame:
    cfg = _cfg()['gcs']
    lo, hi = cfg['valid']
    comps = ', '.join((f"'{c}'" for c in cfg['component_valnames']))
    return con.execute(f"\n        WITH total AS (\n            SELECT c.patientunitstayid AS stay_id,\n                   n.nursingchartoffset AS off,\n                   TRY_CAST(n.nursingchartvalue AS DOUBLE) AS value\n            FROM e_nursecharting n JOIN ecohort c USING (patientunitstayid)\n            WHERE n.nursingchartcelltypevalname = '{cfg['total_valname']}'\n              AND n.nursingchartoffset BETWEEN 0 AND {int(max_hours * 60)}\n        ), parts AS (\n            SELECT c.patientunitstayid AS stay_id, n.nursingchartoffset AS off,\n                   sum(TRY_CAST(n.nursingchartvalue AS DOUBLE)) AS value,\n                   count(DISTINCT n.nursingchartcelltypevalname) AS n_parts\n            FROM e_nursecharting n JOIN ecohort c USING (patientunitstayid)\n            WHERE n.nursingchartcelltypevalname IN ({comps})\n              AND n.nursingchartoffset BETWEEN 0 AND {int(max_hours * 60)}\n            GROUP BY 1, 2\n        ), merged AS (\n            SELECT stay_id, off, value FROM total WHERE value IS NOT NULL\n            UNION ALL\n            SELECT p.stay_id, p.off, p.value FROM parts p\n            LEFT JOIN total t ON t.stay_id = p.stay_id AND t.off = p.off\n            WHERE p.n_parts = 3 AND t.stay_id IS NULL\n        )\n        SELECT stay_id, off / {MINUTES} AS hours, value\n        FROM merged WHERE value BETWEEN {lo} AND {hi}\n        ").pl()

def urine_output(con: duckdb.DuckDBPyConnection, max_hours: float=96.0) -> pl.DataFrame:
    spec = _cfg()['output']['urine']
    like = ' OR '.join((f"lower(i.celllabel) LIKE '{p}'" for p in spec['like']))
    excl = ' AND '.join((f"lower(i.celllabel) NOT LIKE '{p}'" for p in spec.get('exclude_like', []))) or '1=1'
    lo, hi = spec['valid']
    return con.execute(f'\n        SELECT c.patientunitstayid AS stay_id,\n               i.intakeoutputoffset / {MINUTES} AS hours,\n               sum(i.cellvaluenumeric) AS value\n        FROM e_intakeoutput i JOIN ecohort c USING (patientunitstayid)\n        WHERE ({like}) AND ({excl})\n          AND i.cellvaluenumeric BETWEEN {lo} AND {hi}\n          AND i.intakeoutputoffset BETWEEN 0 AND {int(max_hours * 60)}\n        GROUP BY 1, 2\n        ').pl()

def weights(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    lo, hi = _cfg()['weight']['valid']
    return con.execute(f'\n        WITH w AS (\n            SELECT patientunitstayid AS stay_id, gender,\n                   CASE WHEN weight_kg BETWEEN {lo} AND {hi} THEN weight_kg END AS weight_kg\n            FROM ecohort\n        ), med AS (SELECT gender, median(weight_kg) AS fallback FROM w GROUP BY gender)\n        SELECT w.stay_id,\n               coalesce(w.weight_kg, med.fallback) AS weight_kg,\n               w.weight_kg IS NULL AS weight_imputed\n        FROM w LEFT JOIN med USING (gender)\n        ').pl()

def ventilated_hours(con: duckdb.DuckDBPyConnection, max_hours: float=96.0) -> pl.DataFrame:
    return con.execute(f'\n        SELECT DISTINCT c.patientunitstayid AS stay_id,\n               TRY_CAST(r.respchartoffset AS INTEGER) / {MINUTES} AS hours\n        FROM e_respiratorycharting r JOIN ecohort c USING (patientunitstayid)\n        WHERE TRY_CAST(r.respchartoffset AS INTEGER) BETWEEN 0 AND {int(max_hours * 60)}\n        ').pl()

def vasopressor_rates(con: duckdb.DuckDBPyConnection, max_hours: float=96.0) -> pl.DataFrame:
    cfg = _cfg()['infusion']
    cases = ' '.join(('WHEN ' + ' OR '.join((f"lower(i.drugname) LIKE '{p}'" for p in pats)) + f" THEN '{drug}'" for drug, pats in cfg.items()))
    any_like = ' OR '.join((f"lower(i.drugname) LIKE '{p}'" for pats in cfg.values() for p in pats))
    return con.execute(f'\n        SELECT DISTINCT c.patientunitstayid AS stay_id,\n               CASE {cases} END AS drug,\n               TRY_CAST(i.infusionoffset AS INTEGER) / {MINUTES} AS hours,\n               CAST(NULL AS DOUBLE) AS rate\n        FROM e_infusiondrug i JOIN ecohort c USING (patientunitstayid)\n        WHERE ({any_like})\n          AND TRY_CAST(i.infusionoffset AS INTEGER) BETWEEN 0 AND {int(max_hours * 60)}\n        ').pl()

def culture_times(con: duckdb.DuckDBPyConnection, max_hours: float=96.0) -> pl.DataFrame:
    spec = _cfg()['culture']
    like = ' OR '.join((f"lower(t.treatmentstring) LIKE '{p}'" for p in spec['like']))
    return con.execute(f'\n        SELECT DISTINCT c.patientunitstayid AS stay_id,\n               t.treatmentoffset / {MINUTES} AS hours\n        FROM e_treatment t JOIN ecohort c USING (patientunitstayid)\n        WHERE ({like}) AND t.treatmentoffset BETWEEN 0 AND {int(max_hours * 60)}\n        ').pl()

def antibiotic_times(con: duckdb.DuckDBPyConnection, max_hours: float=96.0) -> pl.DataFrame:
    spec = _cfg()['antibiotics']
    like = ' OR '.join((f"lower(t.treatmentstring) LIKE '{p}'" for p in spec['like']))
    return con.execute(f'\n        SELECT DISTINCT c.patientunitstayid AS stay_id,\n               t.treatmentoffset / {MINUTES} AS hours\n        FROM e_treatment t JOIN ecohort c USING (patientunitstayid)\n        WHERE ({like}) AND t.treatmentoffset BETWEEN 0 AND {int(max_hours * 60)}\n        ').pl()

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

Predictions = dict[str, dict[str, np.ndarray]]

TRANSFERRED, ORACLE = (0, 1)

MAX_NONFINITE_SHARE = 0.05

def BootstrapResult(point, se, draws):
    return {'point': point, 'se': se, 'draws': draws}

def boot_n_nonfinite(b):
    return int((~np.isfinite(b['draws'])).sum())

def boot_ci(b, level):
    good = b['draws'][np.isfinite(b['draws'])]
    dropped = b['draws'].size - good.size
    if dropped and dropped / b['draws'].size > MAX_NONFINITE_SHARE:
        raise ValueError(str(dropped) + ' of ' + str(b['draws'].size) + ' bootstrap replicates were undefined; too few clusters carry events for this estimate')
    alpha = (1 - level) / 2
    lo, hi = np.percentile(good, [100 * alpha, 100 * (1 - alpha)])
    return (float(lo), float(hi))


def _gap_for(y: np.ndarray, block: np.ndarray, idx: np.ndarray, seed_sel: np.ndarray) -> float:
    chosen = block[seed_sel][:, :, idx]
    flat = chosen.reshape(-1, chosen.shape[-1])
    a = auroc_batch(y[idx], flat).reshape(chosen.shape[0], 2)
    return float(np.nanmean(a[:, ORACLE] - a[:, TRANSFERRED]))

def two_level_bootstrap(y: dict[str, np.ndarray], preds: Predictions, label_a: str, label_b: str, n_boot: int=2000, seed: int=0, resample_patients: bool=True, resample_seeds: bool=True, clusters: np.ndarray | None=None) -> BootstrapResult:
    models = sorted(preds[label_a])
    if sorted(preds[label_b]) != models:
        raise ValueError('the two labels were not run on the same model families')
    n_patients = preds[label_a][models[0]].shape[-1]
    n_seeds = preds[label_a][models[0]].shape[0]
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_patients)
    all_seeds = np.arange(n_seeds)

    def did(idx: np.ndarray, seed_sel: dict[str, np.ndarray]) -> float:
        per_model = [_gap_for(y[label_a], preds[label_a][m], idx, seed_sel[m]) - _gap_for(y[label_b], preds[label_b][m], idx, seed_sel[m]) for m in models]
        return float(np.mean(per_model))
    point = did(all_idx, dict.fromkeys(models, all_seeds))
    members: list[np.ndarray] = []
    if clusters is not None:
        if len(clusters) != n_patients:
            raise ValueError('clusters must be one label per patient')
        members = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        if not resample_patients:
            idx = all_idx
        elif members:
            picked = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[p] for p in picked])
        else:
            idx = rng.integers(0, n_patients, n_patients)
        sel = {m: rng.integers(0, n_seeds, n_seeds) if resample_seeds else all_seeds for m in models}
        draws[b] = did(idx, sel)
    return BootstrapResult(point=point, se=float(np.std(draws, ddof=1)), draws=draws)

def decompose(y: dict[str, np.ndarray], preds: Predictions, label_a: str, label_b: str, n_boot: int=500, seed: int=0, clusters: np.ndarray | None=None) -> dict:
    common = dict(y=y, preds=preds, label_a=label_a, label_b=label_b, n_boot=n_boot, seed=seed, clusters=clusters)
    both = two_level_bootstrap(**common)
    patients = two_level_bootstrap(**common, resample_seeds=False)
    seeds = two_level_bootstrap(**common, resample_patients=False)
    added = float(np.hypot(patients['se'], seeds['se']))
    return {'se_both': round(both['se'], 5), 'se_patients_only': round(patients['se'], 5), 'se_seeds_only': round(seeds['se'], 5), 'se_quadrature': round(added, 5), 'ratio_both_to_quadrature': round(both['se'] / added, 3) if added else None, 'seed_share_of_variance': round(seeds['se'] ** 2 / (patients['se'] ** 2 + seeds['se'] ** 2), 3) if added else None}

def pool_disjoint(results: list[BootstrapResult], weights: list[float]) -> BootstrapResult:
    if not results:
        raise ValueError('nothing to pool')
    n_boot = {r['draws'].size for r in results}
    if len(n_boot) != 1:
        raise ValueError('groups were bootstrapped with different replicate counts')
    w = np.asarray(weights, dtype=float)
    if w.size != len(results):
        raise ValueError('one weight per group is required')
    w = w / w.sum()
    point = float(np.dot(w, [r['point'] for r in results]))
    draws = np.sum([wi * r['draws'] for wi, r in zip(w, results, strict=True)], axis=0)
    good = draws[np.isfinite(draws)]
    return BootstrapResult(point=point, se=float(np.std(good, ddof=1)), draws=draws)

def stack_predictions(records: list[dict], n_seeds: int) -> Predictions:
    out: Predictions = {}
    for r in records:
        block = out.setdefault(r['label'], {})
        if r['model'] not in block:
            n = len(r['transferred'])
            block[r['model']] = np.full((n_seeds, 2, n), np.nan)
        block[r['model']][r['seed'], TRANSFERRED] = r['transferred']
        block[r['model']][r['seed'], ORACLE] = r['oracle']
    return out

SESOI = 0.01

def bootstrap_p_two_sided(draws: np.ndarray, null: float=0.0) -> float:
    draws = np.asarray(draws)
    n = draws.size
    below = float((draws <= null).sum()) / n
    p = 2 * min(below, 1 - below)
    return float(min(1.0, max(p, 1.0 / n)))

def holm(pvalues: dict[str, float], alpha: float=0.05) -> dict[str, dict]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = ({}, 0.0)
    for i, (key, p) in enumerate(items):
        adjusted = min(1.0, max(running, (m - i) * p))
        running = adjusted
        out[key] = {'p_raw': round(p, 5), 'p_holm': round(adjusted, 5), 'reject': bool(adjusted < alpha)}
    return out

def verdict(point: float, ci95: tuple[float, float], ci90: tuple[float, float], signs_agree: bool, sesoi: float=SESOI) -> str:
    excludes_zero = ci95[0] > 0 or ci95[1] < 0
    within_sesoi = ci90[0] > -sesoi and ci90[1] < sesoi
    if excludes_zero and abs(point) >= sesoi and signs_agree:
        return 'meaningful drift asymmetry' if point > 0 else 'meaningful asymmetry in the opposite direction'
    if excludes_zero and within_sesoi:
        return 'distinguishable but trivial'
    if within_sesoi:
        return 'practically equivalent'
    return 'inconclusive'

def delta_min(ci90: tuple[float, float]) -> float:
    return float(max(abs(ci90[0]), abs(ci90[1])))

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











PIPELINE_FROZEN_TAG = 'pipeline-frozen'







NEGATIVE_MARKERS = ('no growth', 'negative', 'not isolated', 'none', 'no organism')

def microlab_main() -> None:
    con = duckdb.connect()
    for source, tables in (('eicu', ['microLab', 'patient', 'treatment', 'diagnosis', 'infusionDrug', 'medication']), ('mimic_iv', ['microbiologyevents', 'icustays'])):
        root = parquet_dir(source)
        for t in tables:
            path = root / f'{t}.parquet'
            if path.exists():
                con.execute(f"CREATE OR REPLACE VIEW {source}_{t.lower()} AS SELECT * FROM read_parquet('{path.as_posix()}')")
    report: dict = {}
    cols = [r[0] for r in con.execute('DESCRIBE eicu_microlab').fetchall()]
    report['microlab_columns'] = cols
    print(f'microLab columns: {cols}')
    n_rows = con.execute('SELECT count(*) FROM eicu_microlab').fetchone()[0]
    n_pat = con.execute('SELECT count(DISTINCT patientunitstayid) FROM eicu_microlab').fetchone()[0]
    n_stays = con.execute('SELECT count(*) FROM eicu_patient').fetchone()[0]
    report['eicu'] = {'microlab_rows': n_rows, 'stays_with_any_record': n_pat, 'total_stays': n_stays, 'frac_stays_with_record': round(n_pat / n_stays, 4), 'records_per_stay_with_any': round(n_rows / max(n_pat, 1), 2)}
    print(f'\neICU microLab: {n_rows:,} rows, {n_pat:,} of {n_stays:,} stays ({n_pat / n_stays:.1%}) have any record')
    m_rows = con.execute('SELECT count(*) FROM mimic_iv_microbiologyevents').fetchone()[0]
    m_stays = con.execute('SELECT count(*) FROM mimic_iv_icustays').fetchone()[0]
    report['mimic_iv'] = {'microbiologyevents_rows': m_rows, 'icu_stays': m_stays, 'rows_per_icu_stay': round(m_rows / m_stays, 1)}
    print(f'MIMIC-IV microbiologyevents: {m_rows:,} rows over {m_stays:,} ICU stays ({m_rows / m_stays:.1f} per stay)')
    print('\norganism vocabulary (top 25 by frequency):')
    organisms = con.execute('\n        SELECT organism, count(*) AS n\n        FROM eicu_microlab GROUP BY 1 ORDER BY n DESC LIMIT 25\n        ').fetchall()
    for org, n in organisms:
        print(f'  {str(org):48s} {n:>7,}')
    report['organism_top'] = [{'organism': str(o), 'n': int(n)} for o, n in organisms]
    like = ' OR '.join((f"lower(organism) LIKE '%{m}%'" for m in NEGATIVE_MARKERS))
    n_neg = con.execute(f'SELECT count(*) FROM eicu_microlab WHERE {like}').fetchone()[0]
    report['rows_marked_negative'] = int(n_neg)
    print(f'\nrows whose organism reads as a negative result: {n_neg:,} ({n_neg / max(n_rows, 1):.1%})')
    print('\nculture sites:')
    sites = con.execute('SELECT culturesite, count(*) n FROM eicu_microlab GROUP BY 1 ORDER BY n DESC').fetchall()
    for s, n in sites[:15]:
        print(f'  {str(s):40s} {n:>7,}')
    report['culture_sites'] = [{'site': str(s), 'n': int(n)} for s, n in sites]
    print('\nfallback: treatment strings mentioning cultures')
    treat = con.execute("\n        SELECT treatmentstring, count(*) n\n        FROM eicu_treatment\n        WHERE lower(treatmentstring) LIKE '%culture%'\n           OR lower(treatmentstring) LIKE '%blood culture%'\n        GROUP BY 1 ORDER BY n DESC LIMIT 15\n        ").fetchall()
    for s, n in treat:
        print(f'  {str(s):70s} {n:>7,}')
    report['treatment_culture_strings'] = [{'s': str(s), 'n': int(n)} for s, n in treat]
    n_treat_pat = con.execute("\n        SELECT count(DISTINCT patientunitstayid) FROM eicu_treatment\n        WHERE lower(treatmentstring) LIKE '%culture%'\n        ").fetchone()[0]
    report['stays_with_culture_treatment'] = int(n_treat_pat)
    print(f'\nstays with a culture-mentioning treatment row: {n_treat_pat:,} ({n_treat_pat / n_stays:.1%})')
    print('\nfallback: diagnosis strings mentioning sepsis/infection')
    n_diag = con.execute("\n        SELECT count(DISTINCT patientunitstayid) FROM eicu_diagnosis\n        WHERE lower(diagnosisstring) LIKE '%sepsis%'\n           OR lower(diagnosisstring) LIKE '%infection%'\n        ").fetchone()[0]
    report['stays_with_infection_diagnosis'] = int(n_diag)
    print(f'  stays with an infection or sepsis diagnosis string: {n_diag:,} ({n_diag / n_stays:.1%})')
    verdict = 'POSITIVE-ONLY (suspicion would become proven infection)' if n_neg / max(n_rows, 1) < 0.01 and n_pat / n_stays < 0.25 else 'APPEARS TO INCLUDE NEGATIVES'
    report['verdict'] = verdict
    print(f'\nVERDICT: {verdict}')
    path = PROJECT_ROOT / 'results' / 'eicu' / 'microlab_check.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(f'-> {path}')

MIN_STAYS = 100

THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4)

def cultures_main() -> None:
    con = duckdb.connect()
    root = parquet_dir('eicu')
    for t in ('patient', 'hospital', 'microLab', 'treatment', 'medication', 'infusionDrug', 'lab'):
        path = root / f'{t}.parquet'
        if path.exists():
            con.execute(f"CREATE OR REPLACE VIEW e_{t.lower()} AS SELECT * FROM read_parquet('{path.as_posix()}')")
    con.execute("\n        CREATE OR REPLACE TABLE hosp_stats AS\n        WITH stays AS (\n            SELECT p.patientunitstayid, p.hospitalid, h.region, h.numbedscategory,\n                   h.teachingstatus\n            FROM e_patient p LEFT JOIN e_hospital h USING (hospitalid)\n        ),\n        micro AS (SELECT DISTINCT patientunitstayid FROM e_microlab),\n        cult_treat AS (\n            SELECT DISTINCT patientunitstayid FROM e_treatment\n            WHERE lower(treatmentstring) LIKE '%cultures%'\n        ),\n        abx AS (\n            SELECT DISTINCT patientunitstayid FROM e_treatment\n            WHERE lower(treatmentstring) LIKE '%antibacterial%'\n               OR lower(treatmentstring) LIKE '%antibiotic%'\n        ),\n        creat AS (\n            SELECT DISTINCT patientunitstayid FROM e_lab\n            WHERE lower(labname) LIKE 'creatinine%'\n        )\n        SELECT s.hospitalid, any_value(s.region) AS region,\n               any_value(s.numbedscategory) AS beds,\n               any_value(s.teachingstatus) AS teaching,\n               count(*) AS n_stays,\n               avg(CASE WHEN m.patientunitstayid IS NOT NULL THEN 1.0 ELSE 0 END)\n                   AS frac_microlab,\n               avg(CASE WHEN c.patientunitstayid IS NOT NULL THEN 1.0 ELSE 0 END)\n                   AS frac_culture_treatment,\n               avg(CASE WHEN a.patientunitstayid IS NOT NULL THEN 1.0 ELSE 0 END)\n                   AS frac_antibiotic,\n               avg(CASE WHEN cr.patientunitstayid IS NOT NULL THEN 1.0 ELSE 0 END)\n                   AS frac_creatinine\n        FROM stays s\n        LEFT JOIN micro m       ON m.patientunitstayid = s.patientunitstayid\n        LEFT JOIN cult_treat c  ON c.patientunitstayid = s.patientunitstayid\n        LEFT JOIN abx a         ON a.patientunitstayid = s.patientunitstayid\n        LEFT JOIN creat cr      ON cr.patientunitstayid = s.patientunitstayid\n        GROUP BY s.hospitalid\n        ")
    rows = con.execute('SELECT * FROM hosp_stats ORDER BY frac_culture_treatment DESC').pl()
    cols = rows.columns
    print(f"{rows.height} hospitals, {int(rows['n_stays'].sum()):,} stays\n")

    def quantiles(col: str) -> dict:
        v = rows[col].to_numpy()
        return {q: round(float(np.quantile(v, q / 100)), 4) for q in (10, 25, 50, 75, 90, 100)}
    print('distribution across hospitals (unweighted):')
    for col in ('frac_microlab', 'frac_culture_treatment', 'frac_antibiotic', 'frac_creatinine'):
        q = quantiles(col)
        print(f'  {col:24s} p10={q[10]:.3f} p25={q[25]:.3f} median={q[50]:.3f} p75={q[75]:.3f} p90={q[90]:.3f} max={q[100]:.3f}')
    big = rows.filter(rows['n_stays'] >= MIN_STAYS)
    print(f'\nhospitals with >= {MIN_STAYS} stays: {big.height}')
    print(f'\nhow many hospitals clear a culture-charting threshold (and >= {MIN_STAYS} stays):')
    print(f"{'threshold':>10s} {'microLab':>20s} {'treatment cultures':>24s}")
    coverage = []
    for t in THRESHOLDS:
        a = big.filter(big['frac_microlab'] >= t)
        b = big.filter(big['frac_culture_treatment'] >= t)
        coverage.append({'threshold': t, 'hospitals_microlab': a.height, 'stays_microlab': int(a['n_stays'].sum()), 'hospitals_culture_treatment': b.height, 'stays_culture_treatment': int(b['n_stays'].sum())})
        print(f"{t:>10.2f} {a.height:>6d} hosp / {int(a['n_stays'].sum()):>7,} stays {b.height:>7d} hosp / {int(b['n_stays'].sum()):>7,} stays")
    print('\nby region (stay-weighted):')
    reg = con.execute('\n        SELECT region, count(*) AS n_hosp, sum(n_stays) AS n_stays,\n               sum(frac_culture_treatment * n_stays) / sum(n_stays) AS frac_culture,\n               sum(frac_antibiotic * n_stays) / sum(n_stays) AS frac_abx,\n               sum(frac_creatinine * n_stays) / sum(n_stays) AS frac_creat\n        FROM hosp_stats GROUP BY region ORDER BY n_stays DESC\n        ').fetchall()
    print(f"{'region':12s} {'hosp':>5s} {'stays':>8s} {'culture':>8s} {'abx':>7s} {'creat':>7s}")
    for r in reg:
        print(f'{str(r[0]):12s} {r[1]:>5d} {r[2]:>8,} {r[3]:>8.3f} {r[4]:>7.3f} {r[5]:>7.3f}')
    out = {'min_stays': MIN_STAYS, 'n_hospitals': rows.height, 'n_stays': int(rows['n_stays'].sum()), 'distribution': {c: quantiles(c) for c in ('frac_microlab', 'frac_culture_treatment', 'frac_antibiotic', 'frac_creatinine')}, 'threshold_coverage': coverage, 'by_region': [dict(zip(['region', 'n_hosp', 'n_stays', 'frac_culture', 'frac_abx', 'frac_creat'], r, strict=True)) for r in reg], 'per_hospital': rows.to_dicts()}
    path = PROJECT_ROOT / 'results' / 'eicu' / 'culture_by_hospital.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')
    print(f'\n-> {path}   (columns: {cols})')

WINDOW_MIN = 6 * 60

HORIZON_MIN = 48 * 60

MIN_LOS_MIN = 12 * 60

LAB_CONCEPTS = {'creatinine': ['creatinine'], 'bilirubin_total': ['total bilirubin'], 'platelets': ['platelets x 1000'], 'pao2': ['pao2'], 'lactate': ['lactate'], 'wbc': ['wbc x 1000'], 'bun': ['bun'], 'sodium': ['sodium'], 'potassium': ['potassium'], 'bicarbonate': ['bicarbonate', 'hco3'], 'hemoglobin': ['hgb'], 'glucose': ['glucose']}

def cohort_main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='48GB'")
    root = parquet_dir('eicu')
    for t in ('patient', 'hospital', 'lab', 'vitalPeriodic', 'vitalAperiodic', 'nurseCharting', 'intakeOutput', 'infusionDrug', 'respiratoryCharting', 'treatment'):
        path = root / f'{t}.parquet'
        if path.exists():
            con.execute(f"CREATE OR REPLACE VIEW e_{t.lower()} AS SELECT * FROM read_parquet('{path.as_posix()}')")
    con.execute(f"\n        CREATE OR REPLACE TABLE ecohort AS\n        WITH ranked AS (\n            SELECT p.patientunitstayid, p.uniquepid, p.hospitalid,\n                   p.unitdischargeoffset, p.gender, p.age,\n                   p.hospitaldischargeyear, p.unittype,\n                   -- age is a string; '> 89' is the de-identified ceiling\n                   CASE WHEN p.age = '> 89' THEN 90\n                        WHEN TRY_CAST(p.age AS INTEGER) IS NOT NULL\n                        THEN TRY_CAST(p.age AS INTEGER) END AS age_years,\n                   ROW_NUMBER() OVER (\n                       PARTITION BY p.uniquepid\n                       ORDER BY p.hospitaldischargeyear, p.patienthealthsystemstayid,\n                                p.unitvisitnumber\n                   ) AS stay_rank\n            FROM e_patient p\n        )\n        SELECT r.*, h.region, h.numbedscategory, h.teachingstatus\n        FROM ranked r LEFT JOIN e_hospital h USING (hospitalid)\n        WHERE r.stay_rank = 1\n          AND r.unitdischargeoffset >= {MIN_LOS_MIN}\n          AND r.age_years >= 18\n        ")
    n_cohort = con.execute('SELECT count(*) FROM ecohort').fetchone()[0]
    n_hosp = con.execute('SELECT count(DISTINCT hospitalid) FROM ecohort').fetchone()[0]
    print(f'eICU cohort: {n_cohort:,} stays across {n_hosp} hospitals')
    attrition = []
    for name, q in (('all unit stays', 'SELECT count(*) FROM e_patient'), ('first stay per patient', 'SELECT count(DISTINCT uniquepid) FROM e_patient'), (f'... and LOS >= {MIN_LOS_MIN // 60} h and age >= 18', 'SELECT count(*) FROM ecohort')):
        attrition.append({'step': name, 'n': con.execute(q).fetchone()[0]})
        print(f"  {name:42s} {attrition[-1]['n']:>8,}")

    def frac_with(sql: str) -> float:
        return float(con.execute(f'\n            SELECT avg(CASE WHEN h.patientunitstayid IS NOT NULL THEN 1.0 ELSE 0 END)\n            FROM ecohort c LEFT JOIN ({sql}) h USING (patientunitstayid)\n            ').fetchone()[0])
    coverage: dict[str, float] = {}
    for concept, names in LAB_CONCEPTS.items():
        like = ' OR '.join((f"lower(labname) LIKE '{n}%'" for n in names))
        coverage[f'lab:{concept}'] = frac_with(f'SELECT DISTINCT patientunitstayid FROM e_lab\n                WHERE ({like}) AND labresult IS NOT NULL\n                  AND labresultoffset >= 0 AND labresultoffset < {WINDOW_MIN}')
    coverage['vital:heart_rate'] = frac_with(f'SELECT DISTINCT patientunitstayid FROM e_vitalperiodic\n            WHERE heartrate IS NOT NULL\n              AND observationoffset >= 0 AND observationoffset < {WINDOW_MIN}')
    coverage['vital:map_invasive'] = frac_with(f'SELECT DISTINCT patientunitstayid FROM e_vitalperiodic\n            WHERE systemicmean IS NOT NULL\n              AND observationoffset >= 0 AND observationoffset < {WINDOW_MIN}')
    coverage['vital:map_noninvasive'] = frac_with(f'SELECT DISTINCT patientunitstayid FROM e_vitalaperiodic\n            WHERE noninvasivemean IS NOT NULL\n              AND observationoffset >= 0 AND observationoffset < {WINDOW_MIN}')
    coverage['vital:spo2'] = frac_with(f'SELECT DISTINCT patientunitstayid FROM e_vitalperiodic\n            WHERE sao2 IS NOT NULL\n              AND observationoffset >= 0 AND observationoffset < {WINDOW_MIN}')
    coverage['chart:gcs'] = frac_with(f"SELECT DISTINCT patientunitstayid FROM e_nursecharting\n            -- eICU names it 'GCS Total', with 'Eyes'/'Motor'/'Verbal'\n            -- alongside; there is no 'Glasgow' string to match on.\n            WHERE nursingchartcelltypevalname IN\n                  ('GCS Total', 'Eyes', 'Motor', 'Verbal')\n              AND nursingchartoffset >= 0 AND nursingchartoffset < {WINDOW_MIN}")
    coverage['output:urine'] = frac_with(f"SELECT DISTINCT patientunitstayid FROM e_intakeoutput\n            WHERE lower(celllabel) LIKE '%urine%'\n              AND intakeoutputoffset >= 0 AND intakeoutputoffset < {WINDOW_MIN}")
    coverage['infusion:vasopressor'] = frac_with(f"SELECT DISTINCT TRY_CAST(patientunitstayid AS BIGINT) AS patientunitstayid\n            FROM e_infusiondrug\n            WHERE (lower(drugname) LIKE '%norepinephrine%'\n                OR lower(drugname) LIKE '%epinephrine%'\n                OR lower(drugname) LIKE '%dopamine%'\n                OR lower(drugname) LIKE '%dobutamine%')\n              AND TRY_CAST(infusionoffset AS INTEGER) BETWEEN 0 AND {HORIZON_MIN}")
    coverage['resp:ventilator'] = frac_with(f'SELECT DISTINCT TRY_CAST(patientunitstayid AS BIGINT) AS patientunitstayid\n            FROM e_respiratorycharting\n            WHERE TRY_CAST(respchartoffset AS INTEGER) BETWEEN 0 AND {HORIZON_MIN}')
    coverage['lab:creatinine_48h'] = frac_with(f"SELECT DISTINCT patientunitstayid FROM e_lab\n            WHERE lower(labname) LIKE 'creatinine%' AND labresult IS NOT NULL\n              AND labresultoffset >= 0 AND labresultoffset <= {HORIZON_MIN}")
    coverage['output:urine_48h'] = frac_with(f"SELECT DISTINCT patientunitstayid FROM e_intakeoutput\n            WHERE lower(celllabel) LIKE '%urine%'\n              AND intakeoutputoffset >= 0 AND intakeoutputoffset <= {HORIZON_MIN}")
    print('\nconcept availability in the cohort:')
    for k, v in sorted(coverage.items(), key=lambda kv: -kv[1]):
        print(f'  {k:28s} {v:.3f}')
    print('\nper-hospital spread of the label-critical concepts:')
    per_hospital = {}
    critical = {'creatinine_48h': f"SELECT DISTINCT patientunitstayid FROM e_lab\n             WHERE lower(labname) LIKE 'creatinine%' AND labresult IS NOT NULL\n               AND labresultoffset >= 0 AND labresultoffset <= {HORIZON_MIN}", 'urine_48h': f"SELECT DISTINCT patientunitstayid FROM e_intakeoutput\n             WHERE lower(celllabel) LIKE '%urine%'\n               AND intakeoutputoffset >= 0 AND intakeoutputoffset <= {HORIZON_MIN}", 'gcs': f"SELECT DISTINCT patientunitstayid FROM e_nursecharting\n             -- eICU names it 'GCS Total', with 'Eyes'/'Motor'/'Verbal'\n            -- alongside; there is no 'Glasgow' string to match on.\n            WHERE nursingchartcelltypevalname IN\n                  ('GCS Total', 'Eyes', 'Motor', 'Verbal')\n               AND nursingchartoffset >= 0 AND nursingchartoffset <= {HORIZON_MIN}", 'platelets': f"SELECT DISTINCT patientunitstayid FROM e_lab\n             WHERE lower(labname) LIKE 'platelets%' AND labresult IS NOT NULL\n               AND labresultoffset >= 0 AND labresultoffset <= {HORIZON_MIN}", 'bilirubin': f"SELECT DISTINCT patientunitstayid FROM e_lab\n             WHERE lower(labname) LIKE 'total bilirubin%' AND labresult IS NOT NULL\n               AND labresultoffset >= 0 AND labresultoffset <= {HORIZON_MIN}", 'culture_treatment': "SELECT DISTINCT patientunitstayid FROM e_treatment\n             WHERE lower(treatmentstring) LIKE '%cultures%'"}
    print(f"{'concept':20s} {'p10':>7s} {'p25':>7s} {'median':>7s} {'p75':>7s} {'p90':>7s}")
    for name, sql in critical.items():
        rows = con.execute(f'\n            SELECT c.hospitalid, count(*) AS n,\n                   avg(CASE WHEN h.patientunitstayid IS NOT NULL THEN 1.0 ELSE 0 END) AS f\n            FROM ecohort c LEFT JOIN ({sql}) h USING (patientunitstayid)\n            GROUP BY c.hospitalid HAVING count(*) >= 100\n            ').fetchall()
        vals = np.array([r[2] for r in rows])
        q = {p: round(float(np.quantile(vals, p / 100)), 3) for p in (10, 25, 50, 75, 90)}
        per_hospital[name] = {'n_hospitals': len(rows), **{f'p{p}': v for p, v in q.items()}}
        print(f'{name:20s} {q[10]:>7.3f} {q[25]:>7.3f} {q[50]:>7.3f} {q[75]:>7.3f} {q[90]:>7.3f}')
    out = {'window_minutes': WINDOW_MIN, 'horizon_minutes': HORIZON_MIN, 'min_los_minutes': MIN_LOS_MIN, 'n_cohort': n_cohort, 'n_hospitals': n_hosp, 'attrition': attrition, 'concept_availability': {k: round(v, 4) for k, v in coverage.items()}, 'per_hospital_spread': per_hospital}
    path = PROJECT_ROOT / 'results' / 'eicu' / 'cohort_concepts.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')
    print(f'\n-> {path}')

MAX_HOURS = 96.0

VITALS = ['heart_rate', 'sbp', 'dbp', 'map', 'resp_rate', 'spo2', 'temperature']

LABS = ['creatinine', 'bilirubin_total', 'platelets', 'pao2', 'lactate', 'wbc', 'bun', 'sodium', 'potassium', 'bicarbonate', 'hemoglobin', 'glucose']

def build_main() -> None:
    t0 = time.time()
    lcfg = load_config('labels/primary.yaml')
    fcfg = load_config('features/physiology_only.yaml')
    task, sofa_cfg = (lcfg['task'], lcfg['sofa'])
    T, H = (task['prediction_hour'], task['horizon_hours'])
    window = int(fcfg['window_hours'])
    con = connect()
    build_cohort(con)
    hospital_quality(con)
    summ = summary(con)
    print(f"hospitals: {summ['n_included_hospitals']}/{summ['n_hospitals']} included, {summ['n_included_stays']:,}/{summ['n_stays']:,} stays")
    print(f"sepsis subset: {summ['n_sepsis_hospitals']} hospitals, {summ['n_sepsis_stays']:,} stays")
    con.execute('\n        CREATE OR REPLACE TABLE ecohort AS\n        SELECT c.* , h.sepsis_subset\n        FROM ecohort c JOIN ehospital h USING (hospitalid)\n        WHERE h.included\n        ')
    register(con, ['lab', 'vitalPeriodic', 'vitalAperiodic', 'nurseCharting', 'intakeOutput', 'infusionDrug', 'respiratoryCharting', 'treatment'])
    cohort = con.execute('SELECT patientunitstayid AS stay_id, hospitalid, region, sepsis_subset FROM ecohort').pl()
    print(f'cohort after inclusion: {cohort.height:,}  [{time.time() - t0:.0f}s]')
    weights_df = weights(con)
    creatinine = lab(con, 'creatinine', MAX_HOURS)
    urine = urine_output(con, MAX_HOURS)
    stays = con.execute(f'SELECT patientunitstayid AS stay_id,\n                   least(unitdischargeoffset / 60.0, {MAX_HOURS}) AS max_hours\n            FROM ecohort').pl()
    sofa = hourly_sofa(SofaInputs(pao2=lab(con, 'pao2', MAX_HOURS), fio2=pl.DataFrame(schema={'stay_id': pl.Int64, 'hours': pl.Float64, 'value': pl.Float64}), ventilated=ventilated_hours(con, MAX_HOURS), platelets=lab(con, 'platelets', MAX_HOURS), bilirubin=lab(con, 'bilirubin_total', MAX_HOURS), mean_arterial_pressure=vital(con, 'map', MAX_HOURS), gcs_total=gcs_total(con, MAX_HOURS), creatinine=creatinine, urine=urine, vasopressors=vasopressor_rates(con, MAX_HOURS)), stays)
    print(f'SOFA: {sofa.height:,} stay-hours  [{time.time() - t0:.0f}s]')
    suspicion = suspicion_of_infection(antibiotic_times(con, MAX_HOURS), culture_times(con, MAX_HOURS))
    aki_cr = first_onset(creatinine_stage(creatinine))
    aki_uo = first_onset(urine_output_stage(urine, weights_df.select('stay_id', 'weight_kg')))
    onsets = {'aki_creatinine': aki_cr, 'sofa_dysfunction': organ_dysfunction_onset(sofa, prediction_hour=T, increase=sofa_cfg['increase'], baseline_rule=sofa_cfg['baseline_rule']), 'aki_urine': aki_uo, 'sepsis3': sepsis3_onset(sofa, suspicion, prediction_hour=T, increase=sofa_cfg['increase'], baseline_rule=sofa_cfg['baseline_rule'])}
    prevalent = {'aki_creatinine': prevalent_from_onsets(aki_cr, T), 'aki_urine': prevalent_from_onsets(aki_uo, T), 'sofa_dysfunction': prevalent_dysfunction(sofa, T, sofa_cfg['increase'], sofa_cfg['prevalence_rule']), 'sepsis3': prevalent_sepsis3(sofa, suspicion, T, sofa_cfg['increase'], rule=sofa_cfg['prevalence_rule'])}
    common, common_summary = common_cohort(cohort.rename({'hospitalid': 'epoch'}).with_columns(epoch=pl.col('epoch').cast(pl.Utf8)), prevalent)
    print(f'common cohort: {common.height:,}  [{time.time() - t0:.0f}s]')
    ev: dict[str, pl.DataFrame] = {}
    for name in VITALS:
        ev[name] = vital(con, name, window)
    for name in LABS:
        ev[name] = lab(con, name, window)
    ev['gcs_total'] = gcs_total(con, window)
    uo_w = urine.join(weights_df.select('stay_id', 'weight_kg'), on='stay_id', how='left')
    ev['urine_rate'] = uo_w.select('stay_id', 'hours', value=pl.col('value') / pl.col('weight_kg'))
    keep = set(common['stay_id'].to_list())
    ev = {k: v.filter(pl.col('stay_id').is_in(keep) & (pl.col('hours') < window)) for k, v in ev.items()}
    assert_no_future_leakage(ev, window)
    variables = VITALS + ['gcs_total'] + LABS + ['urine_rate']
    panel = hourly_panel(ev, common['stay_id'], window)
    static = con.execute("\n        SELECT patientunitstayid AS stay_id, age,\n               gender AS sex,\n               coalesce(unitadmitsource, 'UNKNOWN') AS admission_type\n        FROM ecohort\n        ").pl()
    fm = build(panel, static, variables, list(fcfg['aggregations']), window)
    print(f"features: {fm['tabular'].shape} tabular, {fm['sequence'].shape} sequence [{time.time() - t0:.0f}s]")
    order = pl.DataFrame({'stay_id': fm['stay_ids']})
    y = {}
    for name, o in onsets.items():
        j = order.join(o.select('stay_id', 'onset_hours'), on='stay_id', how='left')
        y[name] = ((j['onset_hours'] > T) & (j['onset_hours'] <= H)).fill_null(False).to_numpy().astype(np.int8)
    meta = order.join(cohort, on='stay_id', how='left')
    hospitals = meta['hospitalid'].to_numpy()
    regions = meta['region'].fill_null('Unknown').to_numpy().astype('U16')
    sepsis_subset = meta['sepsis_subset'].fill_null(False).to_numpy()
    out = PROJECT_ROOT / 'data' / 'processed' / 'eicu_dataset.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, stay_ids=fm['stay_ids'], tabular=fm['tabular'], sequence=fm['sequence'], static=fm['static'], hospitals=hospitals, regions=regions, sepsis_subset=sepsis_subset, tabular_columns=np.array(fm['tabular_columns']), sequence_columns=np.array(fm['sequence_columns']), static_columns=np.array(fm['static_columns']), **{f'y_{k}': v for k, v in y.items()})
    by_region = []
    for r in sorted(set(regions.tolist())):
        m = regions == r
        by_region.append({'region': r, 'n_stays': int(m.sum()), 'n_hospitals': int(len(set(hospitals[m].tolist()))), **{k: round(float(v[m].mean()), 4) for k, v in y.items()}})
    hosp_prev = []
    for h in sorted(set(hospitals.tolist())):
        m = hospitals == h
        hosp_prev.append({'hospitalid': int(h), 'n_stays': int(m.sum()), **{k: round(float(v[m].mean()), 4) for k, v in y.items()}})
    manifest = {'inclusion_rule': {'min_stays_per_hospital': MIN_STAYS_PER_HOSPITAL, 'min_creatinine_coverage': MIN_CREATININE_COVERAGE, 'note': 'data quality only; no threshold on any label component'}, 'sepsis_subset_rule': {'min_culture_rate': SEPSIS_MIN_CULTURE_RATE, 'note': 'secondary analysis only; selects on a label component'}, 'hospitals': summ, 'n_common_cohort': int(fm_n_stays(fm)), 'n_tabular_features': int(fm['tabular'].shape[1]), 'n_sequence_vars': int(fm['sequence'].shape[2]), 'common_cohort_by_hospital': common_summary[:5], 'prevalence_overall': {k: round(float(v.mean()), 4) for k, v in y.items()}, 'prevalence_by_region': by_region, 'prevalence_by_hospital': hosp_prev, 'sepsis_subset_stays': int(sepsis_subset.sum()), 'missing_fraction_tabular': round(float(np.mean(~np.isfinite(fm['tabular']))), 4)}
    path = PROJECT_ROOT / 'results' / 'eicu' / 'dataset_manifest.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('\nprevalence overall:', manifest['prevalence_overall'])
    print('\nby region:')
    for r in by_region:
        print(f"  {r['region']:12s} {r['n_hospitals']:>3d} hosp {r['n_stays']:>7,} " + '  '.join((f'{k}={r[k]:.3f}' for k in y)))
    print(f'\n-> {out}\n-> {path}   [{time.time() - t0:.0f}s]')

power_DELTA_MIN_FACTOR = 2.443

HOLDOUT_FRACTION = 0.3

def power_main() -> None:
    cfg = load_config('experiments/main.yaml')
    sesoi = cfg['sesoi']
    main_run = json.loads((PROJECT_ROOT / 'results' / 'main' / 'main_run.json').read_text('utf-8'))
    manifest = json.loads((PROJECT_ROOT / 'results' / 'eicu' / 'dataset_manifest.json').read_text('utf-8'))
    mimic_recon = json.loads((PROJECT_ROOT / 'results' / 'recon' / 'final_recon.json').read_text('utf-8'))
    ref_epoch = '2014 - 2016'
    ref_prev = next((r for r in mimic_recon['prevalence_on_common_cohort'] if r['epoch'] == ref_epoch))
    references = {}
    for c in cfg['contrasts']:
        est = main_run['estimates'][f"{c['name']}|{ref_epoch}|full"]
        rarer = min(c['label_a'], c['label_b'], key=lambda lab: ref_prev[f'prev_{lab}'])
        n_pos = ref_prev['n'] * ref_prev[f'prev_{rarer}']
        references[c['name']] = {'epoch': ref_epoch, 'se_patients': est['variance_decomposition']['se_patients_only'], 'limiting_label': rarer, 'limiting_prevalence': ref_prev[f'prev_{rarer}'], 'n_positives': round(n_pos)}
        print(f"{c['name']}: reference se={references[c['name']]['se_patients']:.5f} at {round(n_pos):,} positives of {rarer}")
    prev = manifest['prevalence_overall']
    by_region = manifest['prevalence_by_region']
    n_total = manifest['n_common_cohort']
    n_sepsis_subset = manifest['sepsis_subset_stays']
    splits = [{'design': 'random hospital split', 'test_label': f'{int(HOLDOUT_FRACTION * 100)}% of hospitals', 'n_test': int(round(HOLDOUT_FRACTION * n_total))}]
    for r in by_region:
        splits.append({'design': 'leave-one-region-out', 'test_label': r['region'], 'n_test': r['n_stays']})
    print(f"\n{'design':26s} {'held out':14s} {'n_test':>8s} {'contrast':11s} {'n_pos':>7s} {'se':>8s} {'delta_min':>10s}  verdict")
    rows = []
    for s in splits:
        for c in cfg['contrasts']:
            ref = references[c['name']]
            if s['design'] == 'leave-one-region-out':
                region = next((r for r in by_region if r['region'] == s['test_label']))
                p = min(region[c['label_a']], region[c['label_b']])
            else:
                p = min(prev[c['label_a']], prev[c['label_b']])
            n_pos = s['n_test'] * p
            se = ref['se_patients'] * np.sqrt(ref['n_positives'] / max(n_pos, 1))
            dmin = power_DELTA_MIN_FACTOR * se
            verdict = 'equivalence testable' if dmin <= sesoi else 'detection only' if dmin <= 2 * sesoi else 'underpowered'
            rows.append({**s, 'contrast': c['name'], 'limiting_prevalence': round(p, 4), 'n_positives': round(n_pos), 'se': round(float(se), 5), 'delta_min': round(float(dmin), 5), 'verdict': verdict})
            print(f"{s['design']:26s} {str(s['test_label']):14s} {s['n_test']:>8,} {c['name']:11s} {round(n_pos):>7,} {se:>8.5f} {dmin:>10.5f}  {verdict}")
    print(f"\nsecondary sepsis subset: {n_sepsis_subset:,} stays in {manifest['hospitals']['n_sepsis_hospitals']} hospitals")
    ref = references['DiD_sepsis']
    p_sepsis = prev['sepsis3']
    n_pos = HOLDOUT_FRACTION * n_sepsis_subset * p_sepsis
    se = ref['se_patients'] * np.sqrt(ref['n_positives'] / max(n_pos, 1))
    dmin = power_DELTA_MIN_FACTOR * se
    sepsis_row = {'design': 'random hospital split, sepsis subset', 'n_subset': n_sepsis_subset, 'prevalence_sepsis3': round(p_sepsis, 4), 'n_positives_held_out': round(n_pos), 'se': round(float(se), 5), 'delta_min': round(float(dmin), 5), 'verdict': 'equivalence testable' if dmin <= sesoi else 'detection only' if dmin <= 2 * sesoi else 'underpowered'}
    print(f"  Sepsis-3 prevalence {p_sepsis:.4f}; held-out positives {round(n_pos):,}; delta_min {dmin:.5f} -> {sepsis_row['verdict']}")
    out = {'method': 'projected from MIMIC-IV: se scales as 1/sqrt(n_positives) of the rarer label in each contrast; patient term only', 'sesoi': sesoi, 'references': references, 'eicu_prevalence': prev, 'splits': rows, 'sepsis_subset': sepsis_row}
    path = PROJECT_ROOT / 'results' / 'eicu' / 'power_projection.json'
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\n-> {path}')

design_effect_DELTA_MIN_FACTOR = 2.443

def icc_anova(values: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    uniq, inverse = np.unique(clusters, return_inverse=True)
    k = uniq.size
    n = values.size
    sizes = np.bincount(inverse)
    grand = values.mean()
    cluster_means = np.bincount(inverse, weights=values) / sizes
    ss_between = float(np.sum(sizes * (cluster_means - grand) ** 2))
    ss_within = float(np.sum((values - cluster_means[inverse]) ** 2))
    if k < 2 or n - k < 1:
        return (0.0, float(sizes.mean()))
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    m0 = (n - float(np.sum(sizes ** 2)) / n) / (k - 1)
    if ms_between + (m0 - 1) * ms_within <= 0:
        return (0.0, m0)
    icc = (ms_between - ms_within) / (ms_between + (m0 - 1) * ms_within)
    return (float(max(icc, 0.0)), float(m0))

def design_effect_main() -> None:
    cfg = load_config('experiments/eicu.yaml')
    sesoi = cfg['sesoi']
    patient_proj = json.loads((PROJECT_ROOT / 'results' / 'eicu' / 'power_projection.json').read_text('utf-8'))
    z = np.load(PROJECT_ROOT / 'data' / 'processed' / 'eicu_dataset.npz', allow_pickle=False)
    hospitals, regions = (z['hospitals'], z['regions'].astype(str))
    labels = {k[2:]: z[k] for k in z.files if k.startswith('y_')}
    print(f"{'split':16s} {'label':17s} {'hosp':>5s} {'m_bar':>7s} {'ICC':>7s} {'DEFF':>7s} {'sqrt':>6s}")
    rows = []

    def analyse(name: str, mask: np.ndarray) -> dict:
        out = {'split': name, 'n_stays': int(mask.sum()), 'n_hospitals': int(len(np.unique(hospitals[mask]))), 'labels': {}}
        for label, y in labels.items():
            icc, m0 = icc_anova(y[mask].astype(float), hospitals[mask])
            deff = 1.0 + (m0 - 1.0) * icc
            out['labels'][label] = {'icc': round(icc, 5), 'm_bar': round(m0, 1), 'deff': round(deff, 3), 'sqrt_deff': round(float(np.sqrt(deff)), 3)}
            print(f"{name:16s} {label:17s} {out['n_hospitals']:>5d} {m0:>7.1f} {icc:>7.4f} {deff:>7.2f} {np.sqrt(deff):>6.2f}")
        return out
    rows.append(analyse('cv (all)', np.ones(len(hospitals), dtype=bool)))
    for region in cfg['designs']['leave_one_region_out']['regions']:
        mask = regions == region
        if mask.sum() >= 500:
            rows.append(analyse(f'loro:{region}', mask))
    print(f'\nre-projected delta_min (SESOI = {sesoi})')
    print(f"{'design':22s} {'contrast':11s} {'patient':>9s} {'sqrt DEFF':>10s} {'clustered':>10s}  verdict")
    reprojected = []
    for s in patient_proj['splits']:
        contrast = next((c for c in cfg['contrasts_primary'] + cfg['contrasts_secondary'] if c['name'] == s['contrast']))
        key = 'cv (all)' if s['design'] == 'random hospital split' else f"loro:{s['test_label']}"
        row = next((r for r in rows if r['split'] == key), None)
        if row is None:
            continue
        pair = (contrast['label_a'], contrast['label_b'])
        rarer = min(pair, key=lambda lab: float(labels[lab].mean()))
        most_clustered = max(pair, key=lambda lab: row['labels'][lab]['sqrt_deff'])
        sq_rare = row['labels'][rarer]['sqrt_deff']
        sq = row['labels'][most_clustered]['sqrt_deff']
        cv_gain = np.sqrt(3.0) if s['design'] == 'random hospital split' else 1.0
        clustered = s['delta_min'] * sq / cv_gain
        verdict = 'equivalence testable' if clustered <= sesoi else 'detection only' if clustered <= 2 * sesoi else 'underpowered'
        reprojected.append({**s, 'rarer_label': rarer, 'sqrt_deff_rarer': sq_rare, 'most_clustered_label': most_clustered, 'sqrt_deff_used': sq, 'cv_gain': round(float(cv_gain), 3), 'delta_min_clustered_conservative': round(float(clustered), 5), 'delta_min_clustered_optimistic': round(float(s['delta_min'] * sq_rare / cv_gain), 5), 'verdict_clustered': verdict})
        label = 'hospital 3-fold CV' if s['design'] == 'random hospital split' else f"LORO {s['test_label']}"
        print(f"{label:22s} {s['contrast']:11s} {s['delta_min']:>9.5f} {sq:>10.2f} {clustered:>10.5f}  {verdict}")
    out = {'method': 'DEFF = 1 + (m_bar - 1) * ICC from the labels alone; the patient projection is multiplied by sqrt(DEFF) and, for the cross-validated design, divided by sqrt(3) for averaging three disjoint hospital groups', 'caveats': ['DEFF describes a mean; the DiD is a difference of AUROC differences', 'where a test set holds few hospitals the cluster bootstrap is wide because there is little to resample, which no DEFF captures'], 'per_split': rows, 'reprojected': reprojected, 'sesoi': sesoi}
    path = PROJECT_ROOT / 'results' / 'eicu' / 'design_effect.json'
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\n-> {path}')

DATASET = PROJECT_ROOT / 'data' / 'processed' / 'eicu_dataset.npz'

CELLS = PROJECT_ROOT / 'data' / 'processed' / 'eicu_cells'

LABELS_NEEDED = ('aki_creatinine', 'aki_urine', 'sofa_dysfunction', 'sepsis3')

_DATA: dict | None = None

def assert_frozen() -> None:
    import subprocess
    try:
        tags = subprocess.run(['git', 'tag', '--list'], capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        tags = []
    for required in ('eicu-frozen-v2', PIPELINE_FROZEN_TAG):
        if required not in tags:
            raise SystemExit(f'{required} is not tagged; the eICU design is not frozen. To reproduce the published results outside the original research repository, rerun with --skip-seal.')

def load() -> dict:
    z = np.load(DATASET, allow_pickle=False)
    fm = FeatureMatrix(stay_ids=z['stay_ids'], tabular=z['tabular'], tabular_columns=list(z['tabular_columns'].astype(str)), sequence=z['sequence'], sequence_columns=list(z['sequence_columns'].astype(str)), static=z['static'], static_columns=list(z['static_columns'].astype(str)))
    return {'fm': fm, 'hospitals': z['hospitals'], 'regions': z['regions'].astype(str), 'sepsis_subset': z['sepsis_subset'], 'y': {k: z[f'y_{k}'] for k in LABELS_NEEDED}}

def hospital_folds(hospitals: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    uniq = np.unique(hospitals)
    rng = np.random.default_rng(seed)
    assignment = dict(zip(uniq, rng.integers(0, n_folds, uniq.size), strict=True))
    return np.array([assignment[h] for h in hospitals], dtype=np.int64)

def splits(data: dict, cfg: dict) -> list[dict]:
    out = []
    cv = cfg['designs']['hospital_cv']
    uniq = np.unique(data['hospitals'])
    rng = np.random.default_rng(cv['seed'])
    assignment = dict(zip(uniq, rng.permutation(uniq.size) % cv['n_folds'], strict=True))
    group = np.array([assignment[h] for h in data['hospitals']])
    for k in range(cv['n_folds']):
        out.append({'design': 'cv', 'name': f'fold{k}', 'pool_into': 'cv', 'test_mask': group == k})
    for region in cfg['designs']['leave_one_region_out']['regions']:
        mask = data['regions'] == region
        if mask.sum() >= 500:
            out.append({'design': 'loro', 'name': region, 'pool_into': None, 'test_mask': mask})
    return out

def cell_specs(data: dict, cfg: dict) -> list[dict]:
    out = []
    for split in splits(data, cfg):
        for contrast in cfg['contrasts_primary']:
            for size in cfg['training_sizes'] + cfg['training_sizes_secondary']:
                out.append({'contrast': contrast, 'size': size, 'split': split, 'secondary': False})
        for contrast in cfg['contrasts_secondary']:
            for size in cfg['training_sizes']:
                out.append({'contrast': contrast, 'size': size, 'split': split, 'secondary': True})
    for i, c in enumerate(out):
        c['cell_id'] = i
        c['key'] = f"{('SECONDARY|' if c['secondary'] else '')}{c['contrast']['name']}|{c['split']['design']}:{c['split']['name']}|{c['size']}"
    return out

def _init_worker() -> None:
    global _DATA
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[var] = '1'
    _DATA = load()

def run_cell(spec: dict, cfg: dict, data: dict | None=None, write: bool=True) -> dict | None:
    data = data or _DATA
    fm, y_all = (data['fm'], data['y'])
    split, contrast, size = (spec['split'], spec['contrast'], spec['size'])
    subset = data['sepsis_subset'] if spec['secondary'] else None
    test_mask = split['test_mask'] if subset is None else split['test_mask'] & subset
    train_mask = ~split['test_mask'] if subset is None else ~split['test_mask'] & subset
    if test_mask.sum() < 300 or train_mask.sum() < 300:
        return None
    folds = hospital_folds(data['hospitals'], cfg['n_folds'], cfg['bootstrap']['seed'])
    tuned = load_config('models/tuned.yaml')
    records, gaps = ([], [])
    for model_name in cfg['models']:
        for label in (contrast['label_a'], contrast['label_b']):
            for seed in range(cfg['n_seeds']):
                r = run_transfer(fm=fm, y=y_all[label], train_mask=train_mask, test_mask=test_mask, folds=folds, make_model=lambda s, m=model_name: build_model(m, seed=s, **tuned[m]), label=label, model_name=model_name, seed=seed, train_epoch=f"eicu:train:{split['name']}", test_epoch=f"eicu:{split['design']}:{split['name']}", n_folds=cfg['n_folds'], training_size=size, keep_predictions=True)
                gaps.append(transfer_summary(r))
                records.append({'label': label, 'model': model_name, 'seed': seed, 'transferred': r['predictions']['transferred'], 'oracle': r['predictions']['oracle']})
    preds = stack_predictions(records, n_seeds=cfg['n_seeds'])
    y_test = {lab: y_all[lab][test_mask] for lab in (contrast['label_a'], contrast['label_b'])}
    test_hospitals = data['hospitals'][test_mask]
    common = dict(y=y_test, preds=preds, label_a=contrast['label_a'], label_b=contrast['label_b'], clusters=test_hospitals)
    b = two_level_bootstrap(**common, n_boot=cfg['bootstrap']['n_boot'], seed=cfg['bootstrap']['seed'])
    ci95, ci90 = (boot_ci(b, 0.95), boot_ci(b, 0.9))
    est = {'key': spec['key'], 'contrast': contrast['name'], 'design': split['design'], 'split': split['name'], 'training_size': size, 'secondary': spec['secondary'], 'pool_into': split['pool_into'], 'n_test': int(test_mask.sum()), 'n_train': int(train_mask.sum()), 'n_test_hospitals': int(len(np.unique(test_hospitals))), 'bootstrap_unit': 'hospital', 'point': round(b['point'], 5), 'se': round(b['se'], 5), 'ci95': [round(ci95[0], 5), round(ci95[1], 5)], 'ci90': [round(ci90[0], 5), round(ci90[1], 5)], 'p_raw': bootstrap_p_two_sided(b['draws'][np.isfinite(b['draws'])]), 'delta_min': round(delta_min(ci90), 5), 'n_undefined_draws': boot_n_nonfinite(b), 'variance_decomposition': decompose(**common, n_boot=cfg['bootstrap']['decompose_n_boot'], seed=cfg['bootstrap']['seed']), 'verdict': verdict(b['point'], ci95, ci90, True, cfg['sesoi'])}
    if write:
        CELLS.mkdir(parents=True, exist_ok=True)
        stem = CELLS / f"cell{spec['cell_id']:03d}"
        np.save(stem.with_suffix('.draws.npy'), b['draws'])
        stem.with_suffix('.json').write_text(json.dumps({'est': est, 'gaps': gaps}, indent=2, default=float), encoding='utf-8')
    return est

def merge(cfg: dict) -> None:
    estimates, gaps = ({}, [])
    cells = sorted(CELLS.glob('cell*.json'))
    if not cells:
        raise SystemExit('no cells computed yet')
    loaded = []
    for path in cells:
        payload = json.loads(path.read_text('utf-8'))
        draws = np.load(path.with_suffix('').with_suffix('.draws.npy'))
        loaded.append((payload['est'], draws))
        gaps.extend(payload['gaps'])
    by_group: dict[tuple, list] = {}
    for est, draws in loaded:
        if est['pool_into'] == 'cv':
            by_group.setdefault((est['contrast'], est['training_size'], est['secondary']), []).append((est, draws))
            continue
        b = BootstrapResult(point=est['point'], se=est['se'], draws=draws)
        ci95, ci90 = (boot_ci(b, 0.95), boot_ci(b, 0.9))
        row = {k: v for k, v in est.items() if k not in ('pool_into', 'key')}
        row.update({'ci95': [round(ci95[0], 5), round(ci95[1], 5)], 'ci90': [round(ci90[0], 5), round(ci90[1], 5)], 'p_raw': bootstrap_p_two_sided(draws[np.isfinite(draws)]), 'delta_min': round(delta_min(ci90), 5), 'n_undefined_draws': boot_n_nonfinite(b), 'verdict': verdict(est['point'], ci95, ci90, True, cfg['sesoi'])})
        estimates[est['key']] = row
    for (contrast, size, secondary), members in by_group.items():
        members.sort(key=lambda m: m[0]['split'])
        pooled = pool_disjoint([BootstrapResult(point=e['point'], se=e['se'], draws=d) for e, d in members], weights=[e['n_test'] for e, _ in members])
        ci95, ci90 = (boot_ci(pooled, 0.95), boot_ci(pooled, 0.9))
        key = f"{('SECONDARY|' if secondary else '')}{contrast}|cv|{size}"
        estimates[key] = {'contrast': contrast, 'design': 'hospital_cv', 'split': 'pooled', 'training_size': size, 'secondary': secondary, 'n_test': int(sum((e['n_test'] for e, _ in members))), 'n_folds_pooled': len(members), 'fold_points': [e['point'] for e, _ in members], 'fold_sizes': [e['n_test'] for e, _ in members], 'fold_hospitals': [e['n_test_hospitals'] for e, _ in members], 'bootstrap_unit': 'hospital', 'point': round(pooled['point'], 5), 'se': round(pooled['se'], 5), 'ci95': [round(ci95[0], 5), round(ci95[1], 5)], 'ci90': [round(ci90[0], 5), round(ci90[1], 5)], 'p_raw': bootstrap_p_two_sided(pooled['draws'][np.isfinite(pooled['draws'])]), 'delta_min': round(delta_min(ci90), 5), 'n_undefined_draws': boot_n_nonfinite(pooled), 'verdict': verdict(pooled['point'], ci95, ci90, True, cfg['sesoi'])}
    primary = {k: v['p_raw'] for k, v in estimates.items() if not v['secondary']}
    for k, v in holm(primary, alpha=cfg['multiplicity']['alpha']).items():
        estimates[k]['holm'] = v
    for k, v in estimates.items():
        if v['secondary']:
            v['equivalence_unattainable_declared_in_advance'] = True
    out = {'config': cfg, 'n_estimates': len(estimates), 'estimates': estimates, 'gaps': gaps}
    path = PROJECT_ROOT / 'results' / 'eicu' / 'eicu_run.json'
    path.write_text(json.dumps(out, indent=2, default=float), encoding='utf-8')
    print(f"\n{'key':56s} {'DiD':>9s} {'95% CI':>20s} {'d_min':>8s}  verdict")
    for k, e in sorted(estimates.items()):
        ci = f"[{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}]"
        print(f"{k:56s} {e['point']:>+9.4f} {ci:>20s} {e['delta_min']:>8.4f}  {e['verdict']}")
    print(f'\n-> {path}')

def run_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--merge', action='store_true', help='pool and write the final file')
    ap.add_argument('--verify-cell', type=int, default=None, help='recompute one cell serially and compare with the parallel run')
    ap.add_argument('--skip-seal', action='store_true', help='skip the freeze-tag check; for reproduction outside the original research repository')
    args = ap.parse_args()
    t0 = time.time()
    if args.skip_seal:
        print('seal check skipped (--skip-seal): the freeze tags belong to the original research repository; the frozen settings are embedded in this script')
    else:
        assert_frozen()
    cfg = load_config('experiments/eicu.yaml')
    if args.merge:
        merge(cfg)
        return
    data = load()
    specs = cell_specs(data, cfg)
    print(f"{len(specs)} cells; {fm_n_stays(data['fm']):,} stays, {len(np.unique(data['hospitals']))} hospitals")
    if args.verify_cell is not None:
        spec = specs[args.verify_cell]
        path = CELLS / f"cell{spec['cell_id']:03d}.json"
        parallel = json.loads(path.read_text('utf-8'))['est']
        before = path.stat().st_mtime_ns
        serial = run_cell(spec, cfg, data=data, write=False)
        if path.stat().st_mtime_ns != before:
            raise SystemExit('the stored cell was modified during verification')
        keys = ('point', 'se', 'ci95', 'ci90', 'n_test', 'n_test_hospitals')
        diffs = {k: (parallel[k], serial[k]) for k in keys if parallel[k] != serial[k]}
        print(f"cell {args.verify_cell} ({spec['key']})")
        print(f"  parallel: point={parallel['point']:+.5f} se={parallel['se']:.5f}")
        print(f"  serial:   point={serial['point']:+.5f} se={serial['se']:.5f}")
        if diffs:
            print(f'  DIFFERS in {list(diffs)}: {diffs}')
        else:
            print('  IDENTICAL on point, se, both intervals and test-set sizes')
        raise SystemExit(0 if not diffs else 1)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = {pool.submit(run_cell, s, cfg): s for s in specs}
        for fut in as_completed(futures):
            spec = futures[fut]
            est = fut.result()
            done += 1
            if est is None:
                print(f"  [{done}/{len(specs)}] {spec['key']}: skipped, too small")
                continue
            print(f"  [{done}/{len(specs)}] {spec['key']}: {est['point']:+.4f} [{est['ci95'][0]:+.4f},{est['ci95'][1]:+.4f}] d_min={est['delta_min']:.4f} n={est['n_test']:,} hosp={est['n_test_hospitals']} [{time.time() - t0:.0f}s]")
    merge(cfg)
    print(f'\ntotal {time.time() - t0:.0f}s')

STAGES = {
    'microlab': microlab_main,
    'cultures': cultures_main,
    'cohort': cohort_main,
    'build': build_main,
    'power': power_main,
    'design_effect': design_effect_main,
    'run': run_main,
}

if __name__ == '__main__':
    argv = sys.argv[1:]
    chosen = [argv[0]] if argv and argv[0] in STAGES else list(STAGES)
    rest = argv[1:] if argv and argv[0] in STAGES else argv
    for name in chosen:
        sys.argv = [name, *rest]
        print('== ' + name + ' ==')
        STAGES[name]()
