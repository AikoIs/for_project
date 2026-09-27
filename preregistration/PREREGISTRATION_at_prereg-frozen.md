# Preregistration - *When the Label Drifts*

**Status: FROZEN - 2026-09-19, tag `prereg-frozen`.**

The phase-0 exit gate is complete: the aggregate reconnaissance (`results/recon/`)
and the power analysis on measured cohort sizes and prevalences
(`results/power/power_analysis_real.json`) are done, and the findings are written
up in `docs/PHASE0_FINDINGS.md`. **Nothing in sections 3-11 may change from here.**
Any later change is a deviation, recorded in section 13 with its date and reason.

No model has been fitted to any real data at the time of freezing. Phase-0
reconnaissance was restricted to aggregates (counts, proportions, quantiles) and
contains no model performance estimate of any kind. Three label definitions were
changed during phase 0 the SOFA baseline rule (D015), the ventilation source
(D014) and the culture definition (D017), each on evidence that did not involve
fitting a model, and each recorded before this freeze.

---

## 1. Question and hypothesis

Labels in ICU databases are not observations; they are computed by rules. Those
rules draw on sources that differ in how much clinician behaviour they encode:
physiology (creatinine, SOFA), clinician actions (culture drawn, antibiotic
given), and post-discharge billing codes. Clinician and coder behaviour changes
over time and differs between hospitals, so the same label name denotes
different patient sets in different years.

**H1 (primary).** Holding input features, model architecture, cohort, and
prediction task fixed, models trained on labels defined by clinician actions
lose more discrimination when transferred forward in time than models trained on
labels defined by physiology.

**H2 (replication).** The same ordering holds when the shift is between
hospitals rather than between years (eICU).

The hypothesis concerns the **magnitude** of drift only. The direction of
practice change (whether clinicians culture more or less over time) is not
predicted and is not part of any hypothesis.

---

## 2. What this study is not

* Not a study of feature drift. That axis (input features derived from clinician
  actions age faster than vitals) was established by Yang et al. 2022
  (arXiv:2203.16452) and is explicitly *controlled away* here: all labels are
  predicted from an identical physiological feature matrix, so any shared
  feature drift cancels in the difference-of-differences.
* Not an epidemiological study of sepsis incidence (Rhee et al. 2017).
* Not a claim that any label is "correct". We compare robustness, not validity.

---

## 3. Label ranks - FROZEN BEFORE ANY MODEL RUN

Rank = number of distinct discretionary human decisions that must occur for the
label to fire, counted from the rubric below. Higher rank = more
action-dependent.

Rubric (each satisfied criterion adds one step):
(a) requires a lab/measurement to have been ordered at all;
(b) requires a *repeated* measurement at a specific cadence (documentation
    intensity);
(c) requires a diagnostic action taken because the clinician suspected disease;
(d) requires a therapeutic action taken because the clinician suspected disease;
(e) requires a retrospective administrative coding decision made for
    reimbursement.

| Rank | Label | Criteria met | Level | Primary? |
|------|-------|--------------|-------|----------|
| 1 | **AKI by creatinine** (KDIGO serum-creatinine criterion) | a | stay, timed | yes |
| 2 | **Organ dysfunction** (SOFA increase ≥2 from baseline, no infection criterion) | a | stay, timed | yes |
| 3 | **AKI by urine output** (KDIGO UO criterion) | a, b | stay, timed | yes |
| 4 | **Sepsis-3** (SOFA increase ≥2 **and** suspected infection = culture + antibiotic) | a, c, d | stay, timed | yes |

SOFA baseline for ranks 2 and 4 is the score **at the prediction hour** (D015),
computed from [0, 6] h data only and shared byte-for-byte between the two
labels.
| 5 | **Sepsis by ICD** (Angus; Martin) and **AKI by ICD** | a, c, d, e | hospitalisation, untimed | no - secondary, descriptive |

Rank 5 is descriptive only. Two reasons, both fixed in advance: ICD labels carry
no event time, so the prediction task is formally different; and the ICD-9 →
ICD-10 transition of October 2015 falls inside the epoch range, so their apparent
drift confounds a coding-system artefact with a practice change. They are
reported as an upper bound on drift that *includes* the coding-system artefact,
with the 2015 discontinuity shown explicitly.

### 3.1 The two primary contrasts

Both are difference-of-differences over the drift gap defined in §6:

* **DiD_sepsis = Gap(Sepsis-3) - Gap(SOFA dysfunction).** The cleanest contrast
  in the design: both labels share the identical SOFA trunk; the only difference
  is the conjunction with "culture drawn + antibiotic given". Everything else
  feature drift, case-mix drift, SOFA-component drift is shared and cancels.
* **DiD_AKI = Gap(AKI by urine output) - Gap(AKI by creatinine).** Same KDIGO
  framework, two different criteria.

**Prespecified interpretive caveat on DiD_AKI.** Urine-output AKI is
action-dependent through *documentation intensity* (hourly UO requires a
catheter and a nursing entry), not through a treatment decision. This is a
different mechanism from Sepsis-3. DiD_AKI is therefore co-primary but weaker in
interpretation, and DiD_sepsis is the single headline result.

**Prespecified limitation.** No label here is purely physiological: creatinine
must also be ordered, and the KDIGO baseline-creatinine rule itself depends on
ordering practice. The design measures a *gradient* of action-dependence, not a
physiology-versus-action dichotomy.

---

## 4. Cohort, epochs, and prediction task

* Adults (age ≥ 18 at admission), **first ICU stay per patient only**, ICU
  length of stay ≥ 12 h.
* **Common cohort (primary):** the intersection of patients eligible for all
  four primary labels i.e. patients with no event of *any* primary label
  before the prediction time. Makes the between-label bootstrap exactly paired.
  Per-label cohorts are a prespecified sensitivity analysis.
* Prediction time **T = 6 h** after ICU admission. Features use data from
  [0, 6] h only. Label = event onset in **(6, 48] h**. Patients whose event
  began at or before T are excluded (for the common cohort, excluded everywhere).
* MIMIC-IV epochs from `anchor_year_group`: 2008-10, 2011-13, 2014-16, 2017-19,
  2020-22. Epoch is assigned per patient. Training epochs: 2008-10 + 2011-13.
  Target epochs: 2014-16, 2017-19, 2020-22.
* MIMIC-III CareVue (2001-2008) is a **robustness epoch**, not a primary target,
  because it confounds calendar time with a change of clinical information
  system (CareVue → MetaVision, different `itemid` space, different coverage).
* eICU 2.0 replicates across hospitals, not time (data are 2014-2015 only):
  random hospital splits (primary, 208 hospitals) and leave-one-region-out
  (illustrative, 4 regions).

**Resolved at the phase-0 gate.** UnfoldML (arXiv:2210.15056) reports that 88%
of MIMIC-III sepsis onsets fall in the first 6 h, which would have gutted the
eligible-event count. Measured on MIMIC-IV, that concentration turned out to be
a property of the *baseline convention*, not of sepsis: with the baseline taken
at the prediction hour rather than as a minimum over [0, 6] h, the share of
onsets preceding hour 6 is 0.26 for SOFA dysfunction and 0.27 for Sepsis-3,
in line with the AKI labels (0.29 and 0.20). See D015 for the measured
alternatives.

T = 6 h and the (6, 48] h horizon are therefore **kept as originally
specified**. The common cohort sizes that follow are:

| epoch | cohort | common cohort |
|---|---:|---:|
| 2008-10 | 17,066 | 13,726 |
| 2011-13 | 12,991 | 10,626 |
| 2014-16 | 12,729 | 10,551 |
| 2017-19 | 11,729 | 9,727 |
| 2020-22 | 8,392 | 6,974 |

Training epochs supply 24,352 stays; the target epochs 10,551 / 9,727 / 6,974.
Label prevalence on the common cohort ranges from 0.12-0.15 (Sepsis-3) to
0.40-0.46 (urine AKI); full table in `results/recon/final_recon.json`.

**Prevalent cases.** The two KDIGO labels are states entered once, so a stay is
excluded when its onset falls at or before T. The two rise-based labels are
**deterioration** labels "will this patient become 2 SOFA points worse than
they are now", for which no coherent prevalent case exists, and none is
excluded (D018). Both rise-based labels are treated identically, so DiD_sepsis
still isolates the infection criterion alone. The onset search for those labels
runs strictly after T; searching earlier hours would count a patient who was
sicker at hour 2 and improved by hour 6 as having deteriorated.

---

## 5. Features and models

* **Primary feature set: physiology only.** Vitals, laboratory values, GCS,
  urine output, age, sex, admission type. No measurement-presence indicators, no
  orders, no medications. Identical matrix for every label.
* **Ablation (secondary): physiology + actions.** Adds culture drawn,
  antibiotic given, vasopressor, mechanical ventilation, and measurement-presence
  indicators.
* Models: logistic regression, XGBoost (GPU), GRU (PyTorch). 5 seeds each.
* Hyperparameters are tuned **only within the training epoch**, by internal
  cross-validation, and are then frozen for every transfer and every oracle.
  The oracle uses the same frozen hyperparameters, so the oracle measures
  "retrained on current data", not "retuned on current data".
* Imputation and scaling are fitted on training folds only; enforced by test.

---

## 6. The drift gap, with cross-fitted oracles

For a label L, a model family M, a seed s, and target epoch E:

* **Transferred model**: trained on the training epoch (2008-13), scored on
  *every* patient in E.
* **Oracle**: 5-fold cross-fitting *within* E. For each fold k, a model of the
  same family with the same frozen hyperparameters is trained on the other four
  folds of E and predicts fold k. Concatenating the five held-out folds gives an
  out-of-fold prediction for **every** patient in E.

Both predictions therefore cover 100% of epoch E, on exactly the same patients.

    Gap(L, M, s, E) = AUROC_oracle(L, M, s, E) - AUROC_transferred(L, M, s, E)

The oracle separates "the model has aged" from "the task got harder in this
epoch". Folds are stratified by outcome and split by patient; fold assignment is
shared across all labels and all models within an epoch, so the oracle's
sampling noise is common to the labels being contrasted.

**Oracle training-size note.** Cross-fitting trains each oracle on 80% of E,
which is generally smaller than the training epoch. A learning-curve check
(oracle retrained on subsamples of E) is run once and reported, so that any Gap
attributable to training-set size rather than to era can be read off.

### 6.1 Three scalings of the gap

Reported together for every result. The headline claim is accepted only if the
**sign agrees across all three**.

1. **Absolute** - `Gap` as defined above. *Primary.*
2. **Relative** - `Gap / (AUROC_oracle - 0.5)`, the fraction of attainable
   discrimination lost. Guards against the ceiling effect: an easy label has
   less room to fall.
3. **Logit** - `logit(AUROC_oracle) - logit(AUROC_transferred)`, variance-
   stabilising near the ceiling.

Secondary metrics, reported for all cells: AUPRC gap, scaled Brier score,
calibration slope and intercept, and prevalence of each label.

---

## 7. Primary outcome and estimand

**Primary estimand: the model-averaged DiD_sepsis in each target epoch.**

The hypothesis is about the label, not the model ("The label is to blame, not the model"),
 so the primary estimate averages the per-model DiD across the three
model families, each first averaged over its 5 seeds:

    DiD_sepsis(E) = mean_M [ mean_s Gap(Sepsis-3, M, s, E) - mean_s Gap(SOFA, M, s, E) ]

Per-model DiD values are prespecified secondary outcomes, reported in full.
The same definition applies to DiD_AKI.

**Primary family for multiplicity:** 2 contrasts x 3 target epochs = 6
estimates, Holm-corrected.

---

## 8. Uncertainty

* **Bootstrap:** 2000 replicates, resampling **patients** within the target
  epoch. One resample per replicate is applied simultaneously to every label,
  model, seed, oracle and transferred prediction, so all contrasts are exactly
  paired. Predictions are computed once and reused; no model is refitted inside
  the bootstrap.
* 95% percentile CIs for point estimates; 90% CIs for the equivalence testing in
  §10 (the standard TOST correspondence).
* **What the intervals do and do not cover.** They account for the sampling
  variation of the test patients and for the seed-to-seed variation of model
  fitting, but they are **conditional on the training sample**: the transferred
  model is fitted once, on one draw of the training epochs, and that draw is
  never resampled. An interval here therefore answers "how much would this
  estimate move on a different set of test patients", not "how much would it
  move if the whole study were repeated from a fresh training cohort". The
  latter is strictly wider. The same caveat applies to the cross-fitted oracle,
  whose five fits come from one partition of the target epoch.
* Holm correction within the primary family of 6.
* Effect sizes are reported in preference to bare p-values throughout. This is
  an **estimation study**: the headline result is a point estimate with an
  interval, and a tight interval around zero is a publishable finding.

---

## 9. Smallest effect size of interest (SESOI)

**SESOI = 0.01 absolute AUROC for a DiD.**

Justification, three converging anchors:

1. **Convention.** In the clinical prediction-model literature a difference in
   AUROC below 0.01 is routinely described as negligible; claims of meaningful
   improvement generally start around 0.02. A label-source effect below 0.01
   cannot change what anyone does.
2. **Decision relevance.** The practical recommendation this study could support
   is "if your label depends on clinician actions, revalidate sooner". For that
   to be actionable, the *extra* degradation attributable to the label source
   must be at least as large as the smallest degradation that would prompt
   revalidation of a deployed model, which is of the order of 0.01 AUROC.
3. **Comparison to a known quantity.** Adding or removing one strong predictor
   from a well-specified ICU risk model typically moves AUROC by 0.01-0.03. A
   label-source effect smaller than the weakest such predictor is not of
   interest.

For the relative scaling the corresponding bound is **5% of the oracle's
attainable range** (`0.05 x (AUROC_oracle - 0.5)`), which for a typical oracle
AUROC of 0.80 equals 0.015 - the same order of magnitude, as it should be.

**Noise-floor requirement.** SESOI is only meaningful if it exceeds the
estimator's own jitter. Prespecified check: the seed-to-seed standard deviation
of Gap must be below SESOI/2. If it is not, the number of seeds is increased
until it is, and this is reported.

### 9.1 What the design can actually deliver

Simulated on the frozen design with the measured common-cohort sizes and
prevalences (`results/power/power_analysis_real.json`), cross-fitted oracles:

| contrast | epoch | n | se | expected δ_min at the null | P(equivalent \| truth = 0) | P(meaningful \| truth = 0.02) |
|---|---|---:|---:|---:|---:|---:|
| sepsis | 2014-16 | 10,551 | 0.0032 | 0.0077 | 0.87 | 1.00 |
| sepsis | 2017-19 | 9,727 | 0.0034 | 0.0082 | 0.81 | 1.00 |
| sepsis | 2020-22 | 6,974 | 0.0040 | 0.0097 | 0.62 | 0.99 |
| AKI | 2014-16 | 10,551 | 0.0028 | 0.0067 | 0.95 | 1.00 |
| AKI | 2017-19 | 9,727 | 0.0029 | 0.0070 | 0.94 | 1.00 |
| AKI | 2020-22 | 6,974 | 0.0034 | 0.0084 | 0.79 | 1.00 |

Read honestly. Detecting an effect of 0.02 or more is essentially certain
everywhere. Every expected minimum margin now falls at or below the SESOI of
0.010, so equivalence is testable at the preregistered bound in every cell
though in 2020-22, the smallest epoch, a true null would be certified about
62% of the time for the sepsis contrast rather than the 87% available in
2014-16. That cell remains the weakest and §10's pooled estimate is its
prespecified support.

Power of exactly 0.50 against a true effect of precisely 0.010 is a property of
the decision rule, not a deficiency: once `se < SESOI/1.96` the significance
requirement stops binding and the rule reduces to "point estimate ≥ SESOI",
which an effect of exactly that size clears half the time.

---

## 10. Decision rules, fixed in advance

Evaluated per contrast per epoch, on the Holm-adjusted intervals:

| Outcome | Rule |
|---|---|
| **Meaningful drift asymmetry** | 95% CI excludes 0, point estimate ≥ SESOI, direction as hypothesised, and sign consistent across all three gap scalings |
| **Practically equivalent** | 90% CI lies entirely within ±SESOI (equivalent to TOST at α = 0.05 on both bounds) |
| **Distinguishable but trivial** | 95% CI excludes 0 **and** 90% CI lies within ±SESOI |
| **Inconclusive** | none of the above |

**Equivalence testing (TOST).** Two one-sided bootstrap tests against the bounds
-SESOI and +SESOI. Because a fixed bound can leave an underpowered study
permanently "inconclusive", we additionally report the **minimum equivalence
margin** δ_min = max(|lower 90% bound|, |upper 90% bound|): the smallest margin
at which the data support equivalence. This is reported whatever the outcome, so
the study always yields an interpretable upper bound on the label-source effect
rather than a null verdict.

A significant DiD in the *opposite* direction counts as evidence against H1 and
is reported as such.

**Pooled estimate.** Alongside the three per-epoch estimates, a single pooled
DiD is computed over the union of the target epochs, with the bootstrap
resampling patients within epoch so the pooling respects the epoch structure.
It is a **secondary** outcome the per-epoch estimates remain primary, because
the hypothesis is about ageing and pooling averages that away, but it is the
prespecified route to an equivalence verdict for the 2020-22 sepsis cell
identified as underpowered in §9.1, and it is reported whatever that cell shows,
so it cannot become a result chosen after the fact.

---

## 11. Prespecified secondary and sensitivity analyses

Secondary: ICD labels (descriptive); feature ablation (physiology + actions);
per-model DiD; AUPRC and calibration gaps; CareVue robustness epoch; eICU
replication.

Descriptive: label prevalence, event-onset time distribution, Cohen's between
labels, culture and antibiotic rates all by epoch.

Mechanistic: Spearman correlation between label rank and Gap (5 ranks - power is
very low, reported as descriptive with the ICD point flagged as contaminated);
variance decomposition of Gap into model x label x epoch; shift in the
feature-label association between epochs as a predictor of Gap.

Sensitivity: Sepsis-3 onset variants from Cohen et al. 2024; SOFA baseline rules
`min_before` and `zero`; baseline-creatinine variants; KDIGO stage ≥2 for both
AKI criteria; 24 h and 72 h horizons; prediction hour T = 4; excluding the
2020–22 epoch; **restricting the temporal contrast to 2008-2016** (see §11.1);
per-label cohorts instead of the common cohort; `anchor_year` distance
restriction. The full grid is enumerated in `configs/labels/primary.yaml`.

### 11.1 Label definitions changed during phase 0, before any model was fitted

Three definitions moved between the first draft of this document and the freeze.
All three rest on descriptive reconnaissance counts, proportions, coverage
and none on a model performance estimate, because no model had been fitted.
Recorded here with before-and-after numbers so a reader can judge for themselves.

| change | before | after |
|---|---|---|
| Ventilation source: charted settings, not `procedureevents` (D014) | 2020-22 coverage 0.40x the 2014-16 level | device measurements flat at 0.96-1.00x |
| SOFA baseline: score at the prediction hour, not the minimum over [0, 6] h (D015) | 72-76% of onsets before hour 6; common cohort ~40% | 20-29%; common cohort ~75% |
| Culture definition: diagnostic cultures only, excluding infection-control surveillance (D017) | suspicion 0.62 → 0.38 across epochs; Sepsis-3 0.51 of stays; (SOFA, Sepsis-3) 0.70 → 0.50 | suspicion 0.458 → 0.365; Sepsis-3 0.35; 0.56 → 0.48 |
| Onset searched only after the prediction hour; no prevalent-case exclusion for the rise-based labels (D018) | improvers counted as early events and excluded; common cohort 9,456 / 8,803 / 6,364 | 10,551 / 9,727 / 6,974 |

The culture change (D017) deserves the plainest statement, because it is the one
that touches the headline label. Counting every microbiology specimen the
mimic-code convention made the suspicion rate fall 0.62 → 0.38 across epochs.
41,500 of those specimens are MRSA surveillance screens, concentrated in the
early epochs. A standing surveillance protocol is not a clinician suspecting
infection in a particular patient, so those specimens are excluded, and every
diagnostic culture is kept, urine and sputum included. Sepsis-3 then lands at
0.35 of stays, within reach of published MIMIC-IV cohorts of 16,069-18,661
patients, against 0.51 before.

The permissive `any` variant is retained as a prespecified sensitivity analysis
and reported as a **descriptive result in its own right**: a widely used label
definition acquires substantial apparent drift from an infection-control
programme that has nothing to do with the patients it is applied to.

### 11.2 The culture-rate break, and what it does to the claim

Counting **any** microbiology specimen as a culture, the share of ICU stays with
one in the first 48 h falls from 0.82 to 0.38 across the epochs. That turned out
to be mostly an artefact: 41,500 of those specimens are **MRSA surveillance
screens**, concentrated in the early epochs, and a universal screening programme
is an infection-control policy rather than a clinician suspecting infection in a
particular patient (D017). Excluding surveillance specimens, while keeping
every diagnostic culture, urine and sputum included the culture rate falls
0.479 → 0.353 and the Sepsis-3 count comes into line with published MIMIC-IV
cohorts of 16,069-18,661 patients.

A real decline of about a quarter remains, alongside a near-flat antibiotic rate
(0.627 → 0.594). Its cause cannot be settled inside MIMIC-IV: `poe` has no
microbiology order type, and the `order_subtype` that would carry it is null on
every one of its 8.9 M Lab rows, so no independent trace of the ordering act
exists (D016). Specimens per stay fall less than the share of stays with any
specimen, i.e. cultures concentrate rather than disappear, which is the
signature diagnostic stewardship was designed to produce but is not proof of it.

Three consequences, fixed in advance:

1. The paper states that the proximate cause is not identifiable from these
   data, and does not assert a practice-change mechanism it cannot support.
2. A prespecified sensitivity analysis repeats the primary contrasts using only
   2008-2016, before the break.
3. **The eICU replication is load-bearing, not supporting.** It varies hospital
   at fixed calendar time, so it tests the mechanism labels defined by
   clinician actions transfer worse, with no temporal data-capture confound
   available to explain the result away. H2 is reported alongside H1, not after
   it.

The study's claim survives either cause: a model trained on an action-defined
label degrades once the action process changes, whether that process was bedside
behaviour or data capture. Only the causal story differs.

---

## 12. Data use

PhysioNet credentialed access, DUA in force. Data never leave the local machine.
No patient-level record, and nothing derived from one, enters version control.
Free-text note tables are deliberately not converted. Every aggregate written to
`results/`, a figure, or the paper is suppressed if its cell contains fewer than
20 patients (`min_cell_count` in `configs/paths.yaml`).

---

## 13. Deviations from this preregistration

**2026-09-19 - amended and re-frozen before phase 1.** The document was first
frozen and tagged `prereg-frozen` earlier the same day. Review then found a
defect in the onset logic: the search for a SOFA rise covered hours before the
prediction hour, where, under the at-prediction baseline, an excess means the
patient improved rather than deteriorated. Such patients were recorded as early
events and excluded from the cohort.

The fix and the question it exposed, what a prevalent case even means for a
rise-based label are recorded as D018. Sections 4, 8, 9.1 and 11.1 were
updated, the tag was moved, and the numbers changed as follows: common cohort
10,551 / 9,727 / 6,974 in the target epochs, from 9,456 / 8,803 / 6,364.

Still no model had been fitted to any real data at the time of the amendment,
which is why it is an amendment rather than a deviation from a result. Every
change from here is a deviation and is logged below.
