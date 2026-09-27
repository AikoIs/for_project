# Reproduction code

Code for *Label Provenance and the Temporal Degradation of ICU Prediction
Models: A Preregistered Equivalence Study*. Six scripts run the study end to end,
from the raw PhysioNet CSV files to every figure and result table. All frozen
settings (cohort rules, label definitions, hyperparameters, 15 seeds, 2000
bootstrap replicates, SESOI 0.010) are embedded at the top of each script, so no
separate configuration folder is needed.

| script | what it does |
|---|---|
| `1_extract.py` | raw CSV to parquet, environment check, concept coverage |
| `2_recon.py` | labels, cohort and design, measured on aggregates |
| `3_dataset.py` | one feature matrix, four labels |
| `4_train.py` | MIMIC-IV: tuning, the primary run, the exploratory checks |
| `5_eicu.py` | the replication along the hospital axis |
| `6_figures_and_tables.py` | the seven figures and the result tables |

The preregistration is in `preregistration/`: `PREREGISTRATION.md` is the
current version with every dated amendment and deviation (§13), and
`PREREGISTRATION_at_prereg-frozen.md` is the text as it stood when it was
frozen, before any model was fitted.

## Requirements

**Data.** Credentialed PhysioNet access to both databases (CSV distributions):

- MIMIC-IV v3.1, https://physionet.org/content/mimiciv/3.1/
- eICU-CRD v2.0, https://physionet.org/content/eicu-crd/2.0/

Open `1_extract.py` and set the two paths at the top of the file:

```python
MIMIC_IV_CSV = r"C:/path/to/mimic-iv-3.1"
EICU_CSV = r"C:/path/to/eicu-crd-2.0"
```

**Hardware.** An NVIDIA GPU with CUDA (the GRU and XGBoost run on the GPU, and
`check_env` fails without one). DuckDB is set to 48 GB of memory and 24 threads;
on a smaller machine lower `memory_limit` and `threads` in the `PATHS["duckdb"]`
entry at the top of each script.

**Reproducibility.** Model fitting on the GPU is not bit-for-bit deterministic. A full
rerun of these scripts reproduced every verdict and every Holm decision in the paper;
estimates agreed to within 0.001 and Holm-adjusted p-values to within 0.1.

**Software.** Python 3.11 and:

```
duckdb>=1.1  polars>=1.17  pyarrow>=18  numpy>=2.0  scipy>=1.14
scikit-learn>=1.6  xgboost>=3.0  torch>=2.7  pyyaml>=6.0  matplotlib>=3.9
```

```bash
pip install duckdb polars pyarrow numpy scipy scikit-learn "xgboost>=3.0" pyyaml matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Running

Run the steps below in this order from inside this folder, naming each stage as
shown. Calling a script without a stage name runs every stage it defines in its
internal order, including optional ones (for example `4_train.py` starts with
`tune`, and `2_recon.py` with `power` on its default configuration); that is not
the sequence used for the paper, so name the stages.

**1. Extraction**

```bash
python 1_extract.py convert --source mimic_iv
python 1_extract.py convert --source eicu
python 1_extract.py check_env
python 1_extract.py dictionaries   # optional: lists candidate itemids
python 1_extract.py concepts
```

**2. Labels, cohort and design (aggregates only)**

```bash
python 2_recon.py aki
python 2_recon.py ventilation
python 2_recon.py labels
python 2_recon.py design
python 2_recon.py final        # also writes the measured settings to configs/
python 2_recon.py cultures
python 2_recon.py suspicion
python 2_recon.py power --config experiments/power_real.yaml --out power_analysis_real.json
```

**3. Feature matrix**

```bash
python 3_dataset.py build
python 3_dataset.py smoke      # optional: the smoke run on the 2008-10 -> 2011-13 pair
```

**4. MIMIC-IV analysis**

```bash
python 4_train.py tune            # optional: re-derives the frozen hyperparameters
python 4_train.py seed_variance   # optional: re-derives the seed count
python 4_train.py seed_count      # optional
python 4_train.py main_run --skip-seal   # the 12 preregistered estimates
python 4_train.py report
python 4_train.py learning_curve  # registered in advance: the oracle learning curve
python 4_train.py event_matched   # exploratory
python 4_train.py no_covid        # exploratory
```

The three optional stages repeat how the frozen settings were chosen. They are
not needed to reproduce the results, because the frozen values are already
embedded in the scripts. `tune` writes its selection to `configs/models/tuned.yaml`,
and later stages then read that file instead of the embedded values.

**5. eICU replication**

```bash
python 5_eicu.py microlab
python 5_eicu.py cultures
python 5_eicu.py cohort
python 5_eicu.py build
python 5_eicu.py power
python 5_eicu.py design_effect
python 5_eicu.py run --workers 8 --skip-seal
```

`run` evaluates the cells in parallel and pools them at the end.
`python 5_eicu.py run --merge --skip-seal` re-pools cells that are already computed.

**6. Figures and tables**

```bash
python 6_figures_and_tables.py   # all of its stages (fig01-fig07, tables) are needed, so no stage name
```

## Sealed target epochs

In the original study the target epochs were sealed in code: `4_train.py
main_run` and `5_eicu.py run` refused to evaluate on them until the analysis
had been frozen and tagged in the research repository (`pipeline-frozen`, and
`eicu-frozen-v2` for the replication). The check is kept here so the mechanism
can be read, but those tags belong to the research repository and are not part
of this one. To reproduce the results, pass `--skip-seal` to both commands, as
shown above. It skips only the tag check; every frozen setting is embedded in
the scripts and nothing else changes. What was frozen, and when, is recorded in
`preregistration/`.

## Outputs

Results are written under `output/`, which is excluded from version control; the
exceptions are `2_recon.py final` and `4_train.py tune`, which write the settings they
measure or select to `configs/` (created next to the scripts; a file there takes
precedence over the matching inline `CONFIGS` entry):

- `output/data/`: parquet tables and derived matrices (patient-level; never share)
- `output/results/`: aggregate JSON files per step (`recon/`, `power/`, `smoke/`, `tuning/`, `main/`, `eicu/`)
- `output/results/tables/`: result tables as CSV
- `output/figures/`: the seven figures as PDF and PNG

No patient-level record leaves `output/data/`. Free-text note tables are never
converted.