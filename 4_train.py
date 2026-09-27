from __future__ import annotations
import copy
import sys
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import duckdb
import hashlib
import itertools
import json
import numpy as np
import polars as pl
import time
import warnings
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



def FeatureMatrix(stay_ids, tabular, tabular_columns, sequence, sequence_columns, static, static_columns):
    n = len(stay_ids)
    if not len(tabular) == len(sequence) == len(static) == n:
        raise ValueError('views are not aligned')
    return {'stay_ids': stay_ids, 'tabular': tabular, 'tabular_columns': tabular_columns, 'sequence': sequence, 'sequence_columns': sequence_columns, 'static': static, 'static_columns': static_columns}

def fm_n_stays(fm):
    return len(fm['stay_ids'])

def fm_subset(fm, mask):
    return FeatureMatrix(stay_ids=fm['stay_ids'][mask], tabular=fm['tabular'][mask], tabular_columns=fm['tabular_columns'], sequence=fm['sequence'][mask], sequence_columns=fm['sequence_columns'], static=fm['static'][mask], static_columns=fm['static_columns'])




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
        raise RuntimeError(f'target epochs {sorted(requested)} are sealed until the pipeline is frozen. Tag {PIPELINE_FROZEN_TAG} once features, models and hyperparameters are settled on the {SMOKE_TRAIN_EPOCH} -> {SMOKE_TEST_EPOCH} pair. To reproduce the published results outside the original research repository, rerun with --skip-seal.')

def fold_of(stay_id: int, n_folds: int, seed: int) -> int:
    digest = hashlib.blake2b(f'{seed}:{int(stay_id)}'.encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'big') % n_folds

def assign_folds(stay_ids: np.ndarray, n_folds: int=5, seed: int=0) -> np.ndarray:
    return np.array([fold_of(s, n_folds, seed) for s in stay_ids], dtype=np.int64)





Predictions = dict[str, dict[str, np.ndarray]]

TRANSFERRED, ORACLE = (0, 1)

MAX_NONFINITE_SHARE = 0.05

def BootstrapResult(point, se, draws):
    return {'point': point, 'se': se, 'draws': draws}


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

def expand(space: dict) -> list[dict]:
    grid_keys = space['_grid']
    fixed = {k: v[0] for k, v in space.items() if not k.startswith('_') and k not in grid_keys}
    configs = []
    for combo in itertools.product(*[space[k] for k in grid_keys]):
        cfg = dict(fixed)
        cfg.update(dict(zip(grid_keys, combo, strict=True)))
        configs.append(cfg)
    for extra in space.get('_extra', []):
        cfg = dict(fixed)
        cfg.update({k: space[k][0] for k in grid_keys})
        cfg.update(extra)
        configs.append(cfg)
    seen, out = (set(), [])
    for c in configs:
        key = tuple(sorted(((k, str(v)) for k, v in c.items())))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out

def _clean(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith('_')}

def score_config(ds: Dataset, mask: np.ndarray, folds: np.ndarray, model_name: str, cfg: dict, n_folds: int, seed: int) -> dict:
    fm = fm_subset(dataset_matrix(ds), mask)
    per_label = {}
    for label in LABELS:
        y = dataset_y(ds, label)[mask]
        scores = []
        for k in range(n_folds):
            held = folds == k
            if not held.any() or len(np.unique(y[~held])) < 2:
                continue
            pre = preprocessor_fit(fm_subset(fm, ~held), fitted_on=f'tune|fold:{k}')
            model = build_model(model_name, seed=seed, **_clean(cfg))
            model_fit_matrix(model, fm_subset(fm, ~held), y[~held], pre)
            p = model_predict_matrix(model, fm_subset(fm, held), pre)
            scores.append(auroc(y[held], p))
        per_label[label] = round(float(np.mean(scores)), 4)
    return {'per_label': per_label, 'mean_auroc': round(float(np.mean(list(per_label.values()))), 4)}

def tune_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='logreg,xgboost,gru')
    args = ap.parse_args()
    t0 = time.time()
    space = load_config('models/search_space.yaml')
    cv = space['inner_cv']
    assert_target_epochs_sealed(tuple(cv['epochs']))
    ds = dataset_load(PROJECT_ROOT / 'data' / 'processed' / 'dataset.npz')
    mask = dataset_epoch_mask(ds, *cv['epochs'])
    folds = assign_folds(ds['stay_ids'][mask], cv['n_folds'], seed=cv['fold_seed'])
    print(f"tuning on {mask.sum():,} stays from {cv['epochs']}, {cv['n_folds']}-fold inner CV")
    report, tuned = ({}, {})
    for model_name in args.models.split(','):
        configs = expand(space[model_name])
        print(f'\n{model_name}: {len(configs)} configurations')
        rows = []
        for i, cfg in enumerate(configs):
            r = score_config(ds, mask, folds, model_name, cfg, cv['n_folds'], cv['model_seed'])
            rows.append({'config': _clean(cfg), **r})
            shown = {k: v for k, v in _clean(cfg).items() if k != 'device'}
            print(f"  [{i + 1:2d}/{len(configs)}] mean={r['mean_auroc']:.4f}  {shown}  [{time.time() - t0:.0f}s]")
        best = max(rows, key=lambda r: r['mean_auroc'])
        tuned[model_name] = best['config']
        report[model_name] = {'n_configs': len(rows), 'results': rows, 'selected': best}
        print(f"  -> selected mean_auroc={best['mean_auroc']:.4f}: { {k: v for k, v in best['config'].items() if k != 'device'}}")
    out = CONFIG_ROOT / 'models' / 'tuned.yaml'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('# Selected by 4_train.py tune on the training epochs\n# only, objective = mean inner-CV AUROC across all four labels.\n# Frozen with the pipeline; do not edit by hand.\n' + yaml.safe_dump(tuned, sort_keys=False), encoding='utf-8')
    path = PROJECT_ROOT / 'results' / 'tuning' / 'tuning.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'inner_cv': cv, 'families': report}, indent=2), encoding='utf-8')
    print(f'\n-> {out}\n-> {path}   [{time.time() - t0:.0f}s]')

CONTRASTS = [('DiD_sepsis', 'sepsis3', 'sofa_dysfunction'), ('DiD_aki', 'aki_urine', 'aki_creatinine')]

seed_variance_SESOI = 0.01

def seed_variance_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--models', default='logreg,xgboost,gru')
    ap.add_argument('--training-sizes', default='full,matched')
    ap.add_argument('--n-boot', type=int, default=600)
    args = ap.parse_args()
    t0 = time.time()
    assert_target_epochs_sealed((SMOKE_TRAIN_EPOCH, SMOKE_TEST_EPOCH))
    tuned = load_config('models/tuned.yaml')
    ds = dataset_load(PROJECT_ROOT / 'data' / 'processed' / 'dataset.npz')
    fm = dataset_matrix(ds)
    train_mask, test_mask = (dataset_epoch_mask(ds, SMOKE_TRAIN_EPOCH), dataset_epoch_mask(ds, SMOKE_TEST_EPOCH))
    models = args.models.split(',')
    report = {'seeds_run': args.seeds, 'train_epoch': SMOKE_TRAIN_EPOCH, 'test_epoch': SMOKE_TEST_EPOCH, 'hyperparameters': tuned, 'by_size': {}}
    for size in args.training_sizes.split(','):
        gaps: dict[tuple[str, str, int], float] = {}
        records: list[dict] = []
        for model_name in models:
            for label in LABELS:
                for seed in range(args.seeds):
                    r = run_transfer(fm=fm, y=dataset_y(ds, label), train_mask=train_mask, test_mask=test_mask, folds=ds['folds'], make_model=lambda s, m=model_name: build_model(m, seed=s, **tuned[m]), label=label, model_name=model_name, seed=seed, train_epoch=SMOKE_TRAIN_EPOCH, test_epoch=SMOKE_TEST_EPOCH, training_size=size, keep_predictions=True)
                    gaps[label, model_name, seed] = r['gap']
                    records.append({'label': label, 'model': model_name, 'seed': seed, 'transferred': r['predictions']['transferred'], 'oracle': r['predictions']['oracle']})
            print(f'  {size:7s} {model_name:8s} done [{time.time() - t0:.0f}s]')
        preds = stack_predictions(records, n_seeds=args.seeds)
        y_test = {label: dataset_y(ds, label)[test_mask] for label in LABELS}
        block = {}
        for name, a, b in CONTRASTS:
            per_seed = np.array([float(np.mean([gaps[a, m, s] - gaps[b, m, s] for m in models])) for s in range(args.seeds)])
            sd = float(np.std(per_seed, ddof=1))
            dec = decompose(y_test, preds, a, b, n_boot=args.n_boot)
            se_patients = dec['se_patients_only']
            block[name] = {'did_per_seed_mean': round(float(per_seed.mean()), 5), 'did_per_seed_sd': round(sd, 5), 'decomposition': dec, 'se_seed_component_at': {str(k): round(sd / np.sqrt(k), 5) for k in (5, 10, 15, 20)}, 'total_se_at': {str(k): round(float(np.hypot(se_patients, sd / np.sqrt(k))), 5) for k in (5, 10, 15, 20)}, 'delta_min_at': {str(k): round(2.443 * float(np.hypot(se_patients, sd / np.sqrt(k))), 5) for k in (5, 10, 15, 20)}}
            print(f"\n  [{size}] {name}: DiD per-seed sd={sd:.5f}, se_patients={se_patients:.5f}, se_seeds_only={dec['se_seeds_only']:.5f}, se_both={dec['se_both']:.5f}, seed share of variance={dec['seed_share_of_variance']}")
            for k in (5, 10, 15, 20):
                print(f"      seeds={k:>2d}  total_se={block[name]['total_se_at'][str(k)]:.5f}  delta_min={block[name]['delta_min_at'][str(k)]:.5f}" + ('  <= SESOI' if block[name]['delta_min_at'][str(k)] <= seed_variance_SESOI else ''))
        report['by_size'][size] = block
    smallest = {}
    for size, block in report['by_size'].items():
        for name, b in block.items():
            ok = [int(k) for k, v in b['delta_min_at'].items() if v <= seed_variance_SESOI]
            smallest[f'{size}/{name}'] = min(ok) if ok else None
    report['smallest_seed_count_within_sesoi'] = smallest
    report['recommendation'] = max([v for v in smallest.values() if v is not None], default=None)
    print(f'\nsmallest seed count keeping delta_min <= SESOI: {smallest}')
    print(f"recommended seeds for the main run: {report['recommendation']}")
    path = PROJECT_ROOT / 'results' / 'tuning' / 'seed_variance.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float), encoding='utf-8')
    print(f'\n-> {path}   [{time.time() - t0:.0f}s]')

SEED_GRID = (5, 10, 15, 20)

DELTA_MIN_FACTOR = 2.443

def seed_count_main() -> None:
    cfg = load_config('experiments/main.yaml')
    sesoi = cfg['sesoi']
    measured = json.loads((PROJECT_ROOT / 'results' / 'tuning' / 'seed_variance.json').read_text('utf-8'))
    ds = dataset_load(PROJECT_ROOT / 'data' / 'processed' / 'dataset.npz')
    n_smoke = int(dataset_epoch_mask(ds, measured['test_epoch']).sum())
    sizes = {e: int(dataset_epoch_mask(ds, e).sum()) for e in cfg['target_epochs']}
    print(f"measured on {measured['test_epoch']} (n={n_smoke:,}) with {measured['seeds_run']} seeds")
    for e, n in sizes.items():
        print(f'  {e}: n={n:,}  patient-SE scale x{np.sqrt(n_smoke / n):.3f}')
    cells, per_k = ([], {k: [] for k in SEED_GRID})
    for size, block in measured['by_size'].items():
        for contrast, b in block.items():
            se_p = b['decomposition']['se_patients_only']
            sd_seed = b['did_per_seed_sd']
            for epoch, n in sizes.items():
                se_p_epoch = se_p * np.sqrt(n_smoke / n)
                row = {'contrast': contrast, 'epoch': epoch, 'training_size': size, 'se_patients_scaled': round(float(se_p_epoch), 5), 'sd_seed': sd_seed, 'delta_min_at': {}}
                for k in SEED_GRID:
                    total = float(np.hypot(se_p_epoch, sd_seed / np.sqrt(k)))
                    dmin = DELTA_MIN_FACTOR * total
                    row['delta_min_at'][str(k)] = round(dmin, 5)
                    per_k[k].append(dmin <= sesoi)
                cells.append(row)
    print(f'\nprojected delta_min by cell (SESOI = {sesoi})')
    header = '  '.join((f'k={k:<2d}' for k in SEED_GRID))
    print(f"{'contrast':12s} {'epoch':12s} {'size':8s} {'se_pat':>8s}  {header}")
    for r in cells:
        vals = '  '.join((f"{r['delta_min_at'][str(k)]:.4f}" for k in SEED_GRID))
        print(f"{r['contrast']:12s} {r['epoch']:12s} {r['training_size']:8s} {r['se_patients_scaled']:>8.5f}  {vals}")
    feasible = [k for k in SEED_GRID if all(per_k[k])]
    literal = min(feasible) if feasible else None
    robustness = {}
    for k in SEED_GRID:
        survives = []
        for growth in (1.0, 1.25, 1.5, 2.0):
            ok = True
            for r in cells:
                scale = np.sqrt(n_smoke / sizes[r['epoch']])
                seed_term = r['sd_seed'] * growth * scale / np.sqrt(k)
                total = float(np.hypot(r['se_patients_scaled'], seed_term))
                ok &= DELTA_MIN_FACTOR * total <= sesoi
            survives.append(growth if ok else None)
        robustness[k] = [g for g in survives if g is not None]
    print('\nlargest seed-SD growth factor each seed count tolerates:')
    for k in SEED_GRID:
        worst = max(robustness[k]) if robustness[k] else None
        print(f'  k={k:<2d} tolerates up to x{worst}' if worst else f'  k={k:<2d} fails even with no growth')
    safe = [k for k in feasible if max(robustness[k], default=0) >= 1.5]
    chosen = min(safe) if safe else literal
    print(f"\nall cells within SESOI at: {(feasible if feasible else 'no grid point')}")
    print(f"rule's literal answer: {literal}")
    print(f"smallest k also tolerating a 1.5x seed-SD growth: {(min(safe) if safe else 'none')}")
    if chosen is None:
        worst = max(cells, key=lambda r: r['delta_min_at'][str(max(SEED_GRID))])
        print(f"\nSTOP: even the largest seed count on the grid leaves a cell outside the SESOI. Worst cell: {worst['contrast']} / {worst['epoch']} / {worst['training_size']} -> delta_min={worst['delta_min_at'][str(max(SEED_GRID))]:.5f}")
    else:
        print(f'chosen n_seeds = {chosen} (smallest grid point satisfying every cell)')
    out = {'rule': 'n_seeds = min{k in grid : delta_min <= SESOI in every contrast x target epoch x training size cell}, where se_patients is rescaled to each epoch by sqrt(n_smoke/n_epoch) and delta_min = 2.443 * hypot(se_patients_epoch, sd_seed/sqrt(k))', 'caveat': 'only the patient term is rescaled; the seed term probably grows somewhat in smaller epochs as oracle folds shrink, so these are mildly optimistic. The main run reports the realised decomposition.', 'seed_grid': list(SEED_GRID), 'measured_on': {'epoch': measured['test_epoch'], 'n': n_smoke, 'seeds_run': measured['seeds_run']}, 'target_epoch_sizes': sizes, 'sesoi': sesoi, 'cells': cells, 'feasible_seed_counts': feasible, 'chosen_n_seeds': chosen}
    path = PROJECT_ROOT / 'results' / 'tuning' / 'seed_count_choice.json'
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\n-> {path}')
    if chosen is None:
        raise SystemExit(2)

def main_run_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='check the seal and the config, fit nothing')
    ap.add_argument('--skip-seal', action='store_true', help='skip the freeze-tag check; for reproduction outside the original research repository')
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config('experiments/main.yaml')
    tuned = load_config('models/tuned.yaml')
    if cfg['n_seeds'] is None:
        raise SystemExit('n_seeds is unset: run 4_train.py seed_variance and seed_count, then set n_seeds in the experiments/main.yaml entry of CONFIGS')
    if args.skip_seal:
        print('seal check skipped (--skip-seal): the freeze tags belong to the original research repository; the frozen settings are embedded in this script')
    else:
        assert_target_epochs_sealed(tuple(cfg['target_epochs']))
    if args.dry_run:
        print('seal lifted, config valid; nothing fitted (--dry-run)')
        return
    ds = dataset_load(PROJECT_ROOT / 'data' / 'processed' / 'dataset.npz')
    fm = dataset_matrix(ds)
    train_mask = dataset_epoch_mask(ds, *cfg['train_epochs'])
    n_seeds, boot = (cfg['n_seeds'], cfg['bootstrap'])
    gaps: list[dict] = []
    estimates: dict[str, dict] = {}
    store: dict[str, np.ndarray] = {}
    for epoch in cfg['target_epochs']:
        test_mask = dataset_epoch_mask(ds, epoch)
        y_test = {label: dataset_y(ds, label)[test_mask] for label in LABELS}
        for size in cfg['training_sizes']:
            records = []
            for model_name in cfg['models']:
                for label in LABELS:
                    for seed in range(n_seeds):
                        r = run_transfer(fm=fm, y=dataset_y(ds, label), train_mask=train_mask, test_mask=test_mask, folds=ds['folds'], make_model=lambda s, m=model_name: build_model(m, seed=s, **tuned[m]), label=label, model_name=model_name, seed=seed, train_epoch='+'.join(cfg['train_epochs']), test_epoch=epoch, n_folds=cfg['n_folds'], training_size=size, keep_predictions=True)
                        gaps.append(transfer_summary(r))
                        records.append({'label': label, 'model': model_name, 'seed': seed, 'transferred': r['predictions']['transferred'], 'oracle': r['predictions']['oracle']})
                print(f'  {epoch}  {size:7s} {model_name:8s} done [{time.time() - t0:.0f}s]')
            preds = stack_predictions(records, n_seeds=n_seeds)
            for label in LABELS:
                for model_name in cfg['models']:
                    store[f'{epoch}|{size}|{label}|{model_name}'] = preds[label][model_name]
            for c in cfg['contrasts']:
                key = f"{c['name']}|{epoch}|{size}"
                common = dict(y=y_test, preds=preds, label_a=c['label_a'], label_b=c['label_b'])
                b = two_level_bootstrap(**common, n_boot=boot['n_boot'], seed=boot['seed'])
                ci95, ci90 = (boot_ci(b, 0.95), boot_ci(b, 0.9))
                dec = decompose(**common, n_boot=boot['decompose_n_boot'], seed=boot['seed'])
                sel = [g for g in gaps if g['test_epoch'] == epoch and g['training_size'] == size]
                signs = _sign_agreement(sel, c['label_a'], c['label_b'], cfg['models'])
                estimates[key] = {'contrast': c['name'], 'epoch': epoch, 'training_size': size, 'point': round(b['point'], 5), 'ci95': [round(ci95[0], 5), round(ci95[1], 5)], 'ci90': [round(ci90[0], 5), round(ci90[1], 5)], 'se': round(b['se'], 5), 'p_raw': bootstrap_p_two_sided(b['draws']), 'delta_min': round(delta_min(ci90), 5), 'variance_decomposition': dec, 'sign_agreement_across_scalings': signs, 'verdict': verdict(b['point'], ci95, ci90, signs['all_agree'], cfg['sesoi'])}
                e = estimates[key]
                print(f"    {key}: {e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]  delta_min={e['delta_min']:.4f}  {e['verdict']}")
    adjusted = holm({k: v['p_raw'] for k, v in estimates.items()}, alpha=cfg['multiplicity']['alpha'])
    for k, v in adjusted.items():
        estimates[k]['holm'] = v
    out = {'config': cfg, 'hyperparameters': tuned, 'n_estimates': len(estimates), 'estimates': estimates, 'gaps': gaps, 'elapsed_seconds': round(time.time() - t0, 1)}
    path = PROJECT_ROOT / 'results' / 'main' / 'main_run.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float), encoding='utf-8')
    pred_path = PROJECT_ROOT / 'data' / 'processed' / 'predictions_main.npz'
    np.savez_compressed(pred_path, **store)
    _report(estimates)
    print(f'\n-> {path}\n-> {pred_path}   [{time.time() - t0:.0f}s]')

def _sign_agreement(rows: list[dict], label_a: str, label_b: str, models: list[str]) -> dict:
    out = {}
    for scale in ('gap', 'gap_relative', 'gap_logit'):
        per_model = []
        for m in models:
            a = np.mean([r[scale] for r in rows if r['label'] == label_a and r['model'] == m])
            b = np.mean([r[scale] for r in rows if r['label'] == label_b and r['model'] == m])
            per_model.append(a - b)
        out[scale] = round(float(np.mean(per_model)), 5)
    signs = {np.sign(v) for v in out.values() if v != 0}
    out['all_agree'] = len(signs) <= 1
    return out

def _report(estimates: dict[str, dict]) -> None:
    print(f"\n{'contrast':12s} {'epoch':12s} {'size':8s} {'DiD':>9s} {'95% CI':>20s} {'d_min':>8s} {'p_holm':>8s}  verdict")
    for k, e in estimates.items():
        ci = f"[{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}]"
        print(f"{e['contrast']:12s} {e['epoch']:12s} {e['training_size']:8s} {e['point']:>+9.4f} {ci:>20s} {e['delta_min']:>8.4f} {e['holm']['p_holm']:>8.4f}  {e['verdict']}")

MODELS = ('logreg', 'xgboost', 'gru')

def report_main() -> None:
    data = json.loads((PROJECT_ROOT / 'results' / 'main' / 'main_run.json').read_text('utf-8'))
    est, gaps = (data['estimates'], data['gaps'])
    epochs = data['config']['target_epochs']
    print('=' * 78)
    print('1. THE TWELVE PRIMARY ESTIMATES')
    print('=' * 78)
    print(f"{'contrast':11s} {'epoch':12s} {'size':8s} {'DiD':>8s} {'95% CI':>19s} {'90% CI':>19s} {'d_min':>7s} {'p_holm':>7s}  verdict")
    for e in est.values():
        ci95 = f"[{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}]"
        ci90 = f"[{e['ci90'][0]:+.4f},{e['ci90'][1]:+.4f}]"
        print(f"{e['contrast']:11s} {e['epoch']:12s} {e['training_size']:8s} {e['point']:>+8.4f} {ci95:>19s} {ci90:>19s} {e['delta_min']:>7.4f} {e['holm']['p_holm']:>7.4f}  {e['verdict']}")
    print('\n' + '=' * 78)
    print('2. AUROC LEVELS  (mean over seeds; T = transferred, O = oracle)')
    print('=' * 78)
    for size in data['config']['training_sizes']:
        print(f'\n--- training size: {size} ---')
        header = '  '.join((f'{e[-4:]:>13s}' for e in epochs))
        print(f"{'model':9s} {'label':17s} {header}")
        for model in MODELS:
            for label in LABELS:
                cells = []
                for epoch in epochs:
                    sel = [g for g in gaps if g['model'] == model and g['label'] == label and (g['test_epoch'] == epoch) and (g['training_size'] == size)]
                    t = np.mean([g['auroc_transferred'] for g in sel])
                    o = np.mean([g['auroc_oracle'] for g in sel])
                    cells.append(f'{t:.3f}/{o:.3f}')
                print(f'{model:9s} {label:17s} ' + '  '.join((f'{c:>13s}' for c in cells)))
    print('\n' + '=' * 78)
    print('3. GAP BY LABEL  (oracle - transferred, mean over seeds and models)')
    print('=' * 78)
    for size in data['config']['training_sizes']:
        print(f'\n--- training size: {size} ---')
        header = '  '.join((f'{e[-4:]:>9s}' for e in epochs))
        print(f"{'label':17s} {header}")
        for label in LABELS:
            cells = []
            for epoch in epochs:
                sel = [g for g in gaps if g['label'] == label and g['test_epoch'] == epoch and (g['training_size'] == size)]
                cells.append(f"{np.mean([g['gap'] for g in sel]):+.4f}")
            print(f'{label:17s} ' + '  '.join((f'{c:>9s}' for c in cells)))
    print('\n' + '=' * 78)
    print('4. VARIANCE DECOMPOSITION  (share of variance from seeds)')
    print('=' * 78)
    print(f"{'contrast':11s} {'epoch':12s} {'size':8s} {'se_pat':>8s} {'se_seed':>8s} {'se_both':>8s} {'seed share':>11s} {'both/quad':>10s}")
    for e in est.values():
        d = e['variance_decomposition']
        print(f"{e['contrast']:11s} {e['epoch']:12s} {e['training_size']:8s} {d['se_patients_only']:>8.5f} {d['se_seeds_only']:>8.5f} {d['se_both']:>8.5f} {d['seed_share_of_variance']:>11.3f} {d['ratio_both_to_quadrature']:>10.3f}")
    print('\n' + '=' * 78)
    print('5. SIGN AGREEMENT ACROSS THE THREE GAP SCALINGS')
    print('=' * 78)
    print(f"{'contrast':11s} {'epoch':12s} {'size':8s} {'absolute':>10s} {'relative':>10s} {'logit':>10s}  all agree")
    for e in est.values():
        s = e['sign_agreement_across_scalings']
        print(f"{e['contrast']:11s} {e['epoch']:12s} {e['training_size']:8s} {s['gap']:>+10.5f} {s['gap_relative']:>+10.5f} {s['gap_logit']:>+10.5f}  {('yes' if s['all_agree'] else 'NO')}")
    print('\n' + '=' * 78)
    print('6. DiD_sepsis IN 2020-22, BROKEN DOWN BY MODEL')
    print('=' * 78)
    print(f"{'size':9s} {'model':9s} {'gap sepsis3':>12s} {'gap sofa':>10s} {'DiD':>9s} {'sd over seeds':>14s}")
    for size in data['config']['training_sizes']:
        for model in MODELS:

            def sel(label, m=model, s=size):
                return {r['seed']: r['gap'] for r in gaps if r['test_epoch'] == '2020 - 2022' and r['training_size'] == s and (r['model'] == m) and (r['label'] == label)}
            a, b = (sel('sepsis3'), sel('sofa_dysfunction'))
            did = np.array([a[s] - b[s] for s in sorted(a)])
            print(f'{size:9s} {model:9s} {np.mean(list(a.values())):>+12.4f} {np.mean(list(b.values())):>+10.4f} {did.mean():>+9.4f} {did.std(ddof=1):>14.4f}')
    print("\nevent counts under 'matched', by epoch (derived from the dataset):")
    ds = dataset_load(PROJECT_ROOT / 'data' / 'processed' / 'dataset.npz')
    train_mask = dataset_epoch_mask(ds, *data['config']['train_epochs'])
    n_folds = data['config']['n_folds']
    print(f"{'epoch':12s} {'label':17s} {'train prev':>10s} {'test prev':>10s} {'transf pos':>11s} {'oracle pos':>11s} {'ratio':>6s}")
    for epoch in epochs:
        test_mask = dataset_epoch_mask(ds, epoch)
        n_match = int(round((n_folds - 1) / n_folds * test_mask.sum()))
        for label in LABELS:
            y = dataset_y(ds, label)
            p_tr, p_te = (float(y[train_mask].mean()), float(y[test_mask].mean()))
            transf_pos, oracle_pos = (n_match * p_tr, n_match * p_te)
            print(f'{epoch:12s} {label:17s} {p_tr:>10.4f} {p_te:>10.4f} {transf_pos:>11.0f} {oracle_pos:>11.0f} {transf_pos / oracle_pos:>6.2f}')
    counts: dict[str, int] = {}
    for e in est.values():
        counts[e['verdict']] = counts.get(e['verdict'], 0) + 1
    print('\nverdict tally: ' + ', '.join((f'{k} x{v}' for k, v in sorted(counts.items()))))
    print(f"elapsed: {data['elapsed_seconds']:.0f} s")

FRACTIONS = (0.25, 0.5, 0.75, 1.0)

N_SEEDS = 3

def learning_curve_main() -> None:
    t0 = time.time()
    cfg = load_config('experiments/main.yaml')
    tuned = load_config('models/tuned.yaml')
    ds = dataset_load(PROJECT_ROOT / 'data' / 'processed' / 'dataset.npz')
    fm = dataset_matrix(ds)
    n_folds = cfg['n_folds']
    rows = []
    for epoch in cfg['target_epochs']:
        mask = dataset_epoch_mask(ds, epoch)
        fm_e = fm_subset(fm, mask)
        folds_e = ds['folds'][mask]
        for label in LABELS:
            y_e = dataset_y(ds, label)[mask]
            for model_name in cfg['models']:
                for frac in FRACTIONS:
                    aurocs = []
                    for seed in range(N_SEEDS):
                        rng = np.random.default_rng(7000 + seed)
                        oof = np.full(len(y_e), np.nan)
                        for k in range(n_folds):
                            held = folds_e == k
                            inside = np.flatnonzero(~held)
                            take = rng.choice(inside, size=max(int(round(frac * inside.size)), 50), replace=False)
                            keep = np.zeros(len(y_e), dtype=bool)
                            keep[take] = True
                            if len(np.unique(y_e[keep])) < 2:
                                continue
                            pre = preprocessor_fit(fm_subset(fm_e, keep), 'lc')
                            model = build_model(model_name, seed=seed, **tuned[model_name])
                            model_fit_matrix(model, fm_subset(fm_e, keep), y_e[keep], pre)
                            oof[held] = model_predict_matrix(model, fm_subset(fm_e, held), pre)
                        good = np.isfinite(oof)
                        aurocs.append(auroc(y_e[good], oof[good]))
                    n_train = int(round(frac * (n_folds - 1) / n_folds * mask.sum()))
                    rows.append({'epoch': epoch, 'label': label, 'model': model_name, 'fraction': frac, 'n_oracle_train': n_train, 'n_positives': int(round(frac * (n_folds - 1) / n_folds * y_e.sum())), 'auroc_mean': round(float(np.mean(aurocs)), 4), 'auroc_sd': round(float(np.std(aurocs, ddof=1)), 4)})
                print(f'  {epoch}  {label:17s} {model_name:8s} [{time.time() - t0:.0f}s]')
    out = {'fractions': list(FRACTIONS), 'n_seeds': N_SEEDS, 'preregistered': True, 'rows': rows}
    path = PROJECT_ROOT / 'results' / 'main' / 'learning_curve.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('\nAUROC by oracle training fraction (mean over models and seeds)')
    header = '  '.join((f'{int(f * 100):>6d}%' for f in FRACTIONS))
    print(f"{'epoch':12s} {'label':17s} {header}   slope 75->100%")
    for epoch in cfg['target_epochs']:
        for label in LABELS:
            sel = [r for r in rows if r['epoch'] == epoch and r['label'] == label]
            means = [np.mean([r['auroc_mean'] for r in sel if r['fraction'] == f]) for f in FRACTIONS]
            slope = means[-1] - means[-2]
            flag = '  <-- headline cell' if epoch == '2020 - 2022' and label == 'sepsis3' else ''
            print(f'{epoch:12s} {label:17s} ' + '  '.join((f'{m:.4f}' for m in means)) + f'   {slope:+.4f}{flag}')
    print(f'\n-> {path}   [{time.time() - t0:.0f}s]')

TRAINING_SIZE = 'event_matched'

def event_matched_main() -> None:
    t0 = time.time()
    cfg = load_config('experiments/main.yaml')
    tuned = load_config('models/tuned.yaml')
    ds = dataset_load(PROJECT_ROOT / 'data' / 'processed' / 'dataset.npz')
    fm = dataset_matrix(ds)
    train_mask = dataset_epoch_mask(ds, *cfg['train_epochs'])
    n_seeds, boot = (cfg['n_seeds'], cfg['bootstrap'])
    estimates, gaps = ({}, [])
    for epoch in cfg['target_epochs']:
        test_mask = dataset_epoch_mask(ds, epoch)
        y_test = {label: dataset_y(ds, label)[test_mask] for label in LABELS}
        records = []
        for model_name in cfg['models']:
            for label in LABELS:
                for seed in range(n_seeds):
                    r = run_transfer(fm=fm, y=dataset_y(ds, label), train_mask=train_mask, test_mask=test_mask, folds=ds['folds'], make_model=lambda s, m=model_name: build_model(m, seed=s, **tuned[m]), label=label, model_name=model_name, seed=seed, train_epoch='+'.join(cfg['train_epochs']), test_epoch=epoch, n_folds=cfg['n_folds'], training_size=TRAINING_SIZE, keep_predictions=True)
                    gaps.append(transfer_summary(r))
                    records.append({'label': label, 'model': model_name, 'seed': seed, 'transferred': r['predictions']['transferred'], 'oracle': r['predictions']['oracle']})
            print(f'  {epoch}  {model_name:8s} done [{time.time() - t0:.0f}s]')
        preds = stack_predictions(records, n_seeds=n_seeds)
        for c in cfg['contrasts']:
            key = f"{c['name']}|{epoch}|{TRAINING_SIZE}"
            common = dict(y=y_test, preds=preds, label_a=c['label_a'], label_b=c['label_b'])
            b = two_level_bootstrap(**common, n_boot=boot['n_boot'], seed=boot['seed'])
            ci95, ci90 = (boot_ci(b, 0.95), boot_ci(b, 0.9))
            estimates[key] = {'exploratory': True, 'contrast': c['name'], 'epoch': epoch, 'training_size': TRAINING_SIZE, 'point': round(b['point'], 5), 'ci95': [round(ci95[0], 5), round(ci95[1], 5)], 'ci90': [round(ci90[0], 5), round(ci90[1], 5)], 'se': round(b['se'], 5), 'p_raw': bootstrap_p_two_sided(b['draws']), 'delta_min': round(delta_min(ci90), 5), 'variance_decomposition': decompose(**common, n_boot=boot['decompose_n_boot'], seed=boot['seed']), 'verdict_rule_applied_exploratorily': verdict(b['point'], ci95, ci90, True, cfg['sesoi'])}
            e = estimates[key]
            print(f"    {key}: {e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]  delta_min={e['delta_min']:.4f}")
    for k, v in holm({k: v['p_raw'] for k, v in estimates.items()}).items():
        estimates[k]['holm_within_exploratory_family'] = v
    primary = json.loads((PROJECT_ROOT / 'results' / 'main' / 'main_run.json').read_text('utf-8'))['estimates']
    out = {'exploratory': True, 'training_size': TRAINING_SIZE, 'note': 'not in the frozen pipeline; primary results are unchanged', 'estimates': estimates, 'gaps': gaps, 'elapsed_seconds': round(time.time() - t0, 1)}
    path = PROJECT_ROOT / 'results' / 'main' / 'exploratory_event_matched.json'
    path.write_text(json.dumps(out, indent=2, default=float), encoding='utf-8')
    print('\nEXPLORATORY: event-matched against the frozen variants')
    print(f"{'contrast':11s} {'epoch':12s} {'full':>9s} {'matched':>9s} {'evt-matched':>12s} {'95% CI (evt)':>21s}")
    for c in cfg['contrasts']:
        for epoch in cfg['target_epochs']:
            f = primary[f"{c['name']}|{epoch}|full"]['point']
            m = primary[f"{c['name']}|{epoch}|matched"]['point']
            e = estimates[f"{c['name']}|{epoch}|{TRAINING_SIZE}"]
            ci = f"[{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}]"
            print(f"{c['name']:11s} {epoch:12s} {f:>+9.4f} {m:>+9.4f} {e['point']:>+12.4f} {ci:>21s}")
    print('\nevent counts actually used (mean over seeds, sepsis3):')
    for epoch in cfg['target_epochs']:
        sel = [g for g in gaps if g['label'] == 'sepsis3' and g['test_epoch'] == epoch]
        print(f"  {epoch}: transferred {np.mean([g['n_train_positives'] for g in sel]):.0f} positives, oracle fold {np.mean([g['n_oracle_positives'] for g in sel]):.0f}")
    print(f'\n-> {path}   [{time.time() - t0:.0f}s]')

EPOCH = '2020 - 2022'

COVID_CODES = ('U071', 'U07.1')

def covid_stay_ids() -> set[int]:
    con = connect()
    build_cohort(con)
    register_source(con, 'mimic_iv', ['diagnoses_icd'])
    codes = ', '.join((f"'{c}'" for c in COVID_CODES))
    rows = con.execute(f"""\n        SELECT DISTINCT c.stay_id\n        FROM mimic_iv_diagnoses_icd d\n        JOIN cohort c ON c.hadm_id = d.hadm_id\n        WHERE d.icd_version = 10\n          AND replace(upper(d.icd_code), '.', '') IN\n              ({', '.join((f"'{c.replace('.', '').upper()}'" for c in COVID_CODES))})\n        """).fetchall()
    con.close()
    return {int(r[0]) for r in rows}

def no_covid_main() -> None:
    t0 = time.time()
    cfg = load_config('experiments/main.yaml')
    tuned = load_config('models/tuned.yaml')
    ds = dataset_load(PROJECT_ROOT / 'data' / 'processed' / 'dataset.npz')
    fm = dataset_matrix(ds)
    train_mask = dataset_epoch_mask(ds, *cfg['train_epochs'])
    n_seeds, boot = (cfg['n_seeds'], cfg['bootstrap'])
    covid = covid_stay_ids()
    in_epoch = dataset_epoch_mask(ds, EPOCH)
    is_covid = np.array([int(s) in covid for s in ds['stay_ids']])
    test_mask = in_epoch & ~is_covid
    n_excluded = int((in_epoch & is_covid).sum())
    print(f'{EPOCH}: {in_epoch.sum():,} stays, {n_excluded:,} with U07.1 ({n_excluded / in_epoch.sum():.1%}) excluded -> {test_mask.sum():,}')
    for label in LABELS:
        y = dataset_y(ds, label)
        print(f'  {label:17s} prevalence {y[in_epoch].mean():.4f} -> {y[test_mask].mean():.4f}')
    estimates, gaps = ({}, [])
    y_test = {label: dataset_y(ds, label)[test_mask] for label in LABELS}
    for size in cfg['training_sizes']:
        records = []
        for model_name in cfg['models']:
            for label in LABELS:
                for seed in range(n_seeds):
                    r = run_transfer(fm=fm, y=dataset_y(ds, label), train_mask=train_mask, test_mask=test_mask, folds=ds['folds'], make_model=lambda s, m=model_name: build_model(m, seed=s, **tuned[m]), label=label, model_name=model_name, seed=seed, train_epoch='+'.join(cfg['train_epochs']), test_epoch=f'{EPOCH}|no-covid', n_folds=cfg['n_folds'], training_size=size, keep_predictions=True)
                    gaps.append(transfer_summary(r))
                    records.append({'label': label, 'model': model_name, 'seed': seed, 'transferred': r['predictions']['transferred'], 'oracle': r['predictions']['oracle']})
            print(f'  {size:7s} {model_name:8s} done [{time.time() - t0:.0f}s]')
        preds = stack_predictions(records, n_seeds=n_seeds)
        for c in cfg['contrasts']:
            key = f"{c['name']}|{EPOCH}|no-covid|{size}"
            b = two_level_bootstrap(y=y_test, preds=preds, label_a=c['label_a'], label_b=c['label_b'], n_boot=boot['n_boot'], seed=boot['seed'])
            ci95, ci90 = (boot_ci(b, 0.95), boot_ci(b, 0.9))
            estimates[key] = {'exploratory': True, 'contrast': c['name'], 'epoch': EPOCH, 'training_size': size, 'covid_excluded': n_excluded, 'point': round(b['point'], 5), 'ci95': [round(ci95[0], 5), round(ci95[1], 5)], 'ci90': [round(ci90[0], 5), round(ci90[1], 5)], 'se': round(b['se'], 5), 'p_raw': bootstrap_p_two_sided(b['draws']), 'delta_min': round(delta_min(ci90), 5), 'verdict_rule_applied_exploratorily': verdict(b['point'], ci95, ci90, True, cfg['sesoi'])}
            e = estimates[key]
            print(f"    {key}: {e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]")
    primary = json.loads((PROJECT_ROOT / 'results' / 'main' / 'main_run.json').read_text('utf-8'))['estimates']
    print('\nEXPLORATORY: 2020-22 with and without U07.1 admissions')
    print(f"{'contrast':11s} {'size':8s} {'with covid':>11s} {'without':>11s} {'95% CI (without)':>21s}")
    for c in cfg['contrasts']:
        for size in cfg['training_sizes']:
            was = primary[f"{c['name']}|{EPOCH}|{size}"]['point']
            e = estimates[f"{c['name']}|{EPOCH}|no-covid|{size}"]
            ci = f"[{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}]"
            print(f"{c['name']:11s} {size:8s} {was:>+11.4f} {e['point']:>+11.4f} {ci:>21s}")
    out = {'exploratory': True, 'epoch': EPOCH, 'codes': list(COVID_CODES), 'note': 'U07.1 used only to select patients out; never a feature, never part of a label. Primary results unchanged.', 'n_excluded': n_excluded, 'estimates': estimates, 'gaps': gaps, 'elapsed_seconds': round(time.time() - t0, 1)}
    path = PROJECT_ROOT / 'results' / 'main' / 'exploratory_no_covid.json'
    path.write_text(json.dumps(out, indent=2, default=float), encoding='utf-8')
    print(f'\n-> {path}   [{time.time() - t0:.0f}s]')

STAGES = {
    'tune': tune_main,
    'seed_variance': seed_variance_main,
    'seed_count': seed_count_main,
    'main_run': main_run_main,
    'report': report_main,
    'learning_curve': learning_curve_main,
    'event_matched': event_matched_main,
    'no_covid': no_covid_main,
}

if __name__ == '__main__':
    argv = sys.argv[1:]
    chosen = [argv[0]] if argv and argv[0] in STAGES else list(STAGES)
    rest = argv[1:] if argv and argv[0] in STAGES else argv
    for name in chosen:
        sys.argv = [name, *rest]
        print('== ' + name + ' ==')
        STAGES[name]()
