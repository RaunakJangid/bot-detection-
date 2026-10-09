# SHIELD-IoT v3: final results report

**Runs:** 7–9 October 2026, same desktop as v1/v2 (RTX 3060 12 GB, 20 threads, 32 GB RAM).
**Every table in this report is generated from the experiment output files** by `scripts/21_v3_report.py`
(the template is `results/v3/REPORT_TEMPLATE.md`; nothing is copied by hand). Table CSVs are in `results/v3/tables/`.
Numbers are **test macro-F1 in %** unless a column says otherwise.

## 0. Protocol (unchanged from v2, and applied to everything below)

- **Leakage controls:** `IAT` (CIC datasets) and `seq` (Bot-IoT) removed; exact duplicate feature vectors removed
  across train → validation → test; scaler fitted on train only.
- **Every choice is made on validation data**: thresholds, biases, ensemble weights, calibration, PTQ vs QAT,
  placement settings (on separate tuning graphs). **Test data is scored once**, after all choices are fixed.
- Paired tests are two-sided Wilcoxon signed-rank tests unless stated; "pairs" are task × seed (detection) or
  instances / topologies (placement).
- v1/v2 outputs were only read. v3 outputs: `D:\shield_run\outputs_v3`, full-data runs: `D:\shield_run\outputs_full`.

## 1. Summary of findings

| # | Experiment | Result | Strength of evidence |
|---|---|---|---|
| 1 | Full CICIoT2023 (6.4× more training data after de-dup), **identical test rows** | Same model (ensemble + rule): 34-class 70.98 → **73.22**, 8-class 74.60 → **77.01**. Best subset model of any kind on 34-class: 71.25 | Direct, row-matched; one GBDT seed in the ensemble |
| 2 | Validation-tuned per-class decision bias | Teacher +0.98 pts (35/40 better, p = 1.6e-9); SHIELD +0.67 (26/40, p = 1.5e-4); mixed on CICIoMT (students slightly worse on binary and 6-class) | Significant, small |
| 3 | Integer students (calibration chosen on validation, QAT when needed) | Mean int8 loss 14.93 → **1.05** pts; SHIELD int8 models 3.7–16.4 KB | Strong |
| 4 | Student → expert cascade | CICIoT 34-class: SHIELD 65.71 → 70.29 with 10% of flows escalated (ensemble expert 71.27) | Strong; thresholds set on validation |
| 5 | Cascade drives controller capacity | Same controller count as the student alone, accuracy close to the teacher | Model-based (M/M/1) |
| 6 | SHAP attribution loss | **More faithful explanations** (33/40, p = 2.3e-4), **no accuracy gain** | Significant; one task reversed |
| 7 | Stronger teacher (ResMLP×5 + XGBoost) | Teacher better on 7/8 tasks, **students distilled from it are not better** (all p ≥ 0.12) | Negative result |
| 8 | Placement: 18 methods, exact optima on 37 instances | No method dominates. GWO+SA has the best topology-level rank; the diversity-controlled hybrid has the best instance-level rank; the original EHO–ACO converges prematurely | Mixed; see §7 |
| 9 | Attack-surge isolation | Isolation removes attack exposure (0%) but costs benign throughput except at the largest surge | Model-based |
| 10 | Unseen-attack detection | Student catches 44–49% of unseen Web/BruteForce at 1% FPR; Mahalanobis/FlyHash novelty scores **do not help** | Negative result |

## 2. How much is left to gain? (identifiability)

1-NN error on validation vs the training set (Cover & Hart bounds on the Bayes error). Exact duplicate vectors with
conflicting labels are rare (about 0.004% of test rows, all DoS ↔ DDoS), so the exact ceiling is uninformative; the
neighbourhood estimate is not.

{{ceiling}}

**Reading:** on CICIoT, XGBoost's error (11.8%) is within about 3 points of the lower bound (9.0%). The largest
overlaps are DoS vs DDoS of the same protocol (for example DoS-UDP ↔ DDoS-UDP) and the rare web/brute-force classes
(1-NN F1 0–21%). These bounds are asymptotic estimates, not exact limits. CICIoMT's test set comes from different
captures, so its test error is not directly comparable with the validation-based bound.

## 3. More training data: full CICIoT2023 (the largest real gain)

**Design (directly comparable):** validation and test are **exactly** the strict CICIoT rows used in v1–v3
(re-derived with the same de-dup code and verified label-for-label by `scripts/20_verify_full_split.py`). Training
uses all 169 merged CSVs (46.7M rows) after removing exact duplicates (18.9M), label-conflicting vectors (1.0M) and
**every vector that occurs in the validation or test files (1.9M)**, leaving 25.8M rows (6.4× the subset). Student
settings (k, size) are reused from the subset's validation tuning.

Learning curve measured beforehand (XGBoost, validation, subset fractions):

{{learning_curve}}

The power-law fit predicted +3.9 macro-F1 for 9× data. The full data gives 6.4× after de-dup; the measured test
gain is below:

{{full}}

- **34-class: 73.22** (ensemble + rule) on the identical test rows where the same model trained on the subset gave
  70.98 (and the best subset model of any kind, step 1's ensemble + rule, gave 71.25). XGBoost alone: 68.56 → 70.81.
- **The tiny student gains less** (SHIELD + rule 65.65 → 67.47 on 34-class; 69.49 → 69.60 on 8-class): it is limited
  by capacity more than data. The cascade (SHIELD → ensemble) reaches 72.68 at 20% escalation.
- Limitation: CICIoT2023's merged files have no capture identifiers, so a capture-level split (as in CICIoMT2024) is
  impossible. Exact duplicates are removed; near-duplicates within a capture cannot be. This applies to every
  published CICIoT2023 result too.

### Comparison with published results (indicative only)

| Task | **This work, strict, full data** | Published | Protocol of the published number |
|---|---|---|---|
| CICIoT 34-class | **73.22** | 71.4 (RF, Neto et al., Sensors 2023) | keeps `IAT`, no cross-split de-dup |
| | | 71.1 (LightGBM, H. M. Kim, manuscript under review 2026) | `IAT` removed; own split |
| CICIoT 8-class | **77.01** | 71.9 (RF, Neto et al. 2023) | keeps `IAT` |
| CICIoT binary | 95.70 | 96.5 (RF, Neto et al. 2023) | keeps `IAT` — **we do not beat this** |
| CICIoMT 19-class | 71.60 (ensemble) | 52.2–55.1 (RF, Dadkhah et al., IoT 2024; two secondary sources disagree) | dataset paper baseline; later papers report much higher scores under other protocols |

Two leaderboards (published headline scores vs results reproduced under a comparable protocol on the same test rows):

{{leaderboards}}

The published CICIoMT headline (two-stage CatBoost, 90.39) is reproduced only when the leaky `IAT` feature is kept (92.12); under the strict protocol the same method scores 78.13 on 6-class and 62.01 on 19-class, below ours. These published numbers use **different test sets** (and mostly a leaky feature), so the comparison is indicative;
only the subset → full comparison above is row-matched. We do **not** claim state of the art on CICIoMT2024: later
studies report far higher scores under different protocols, and a strict CICIoMT2024 reference was not found.

## 4. Better models on the original data (smaller gains)

### 4.1 Decision rules fitted on validation (logit adjustment + per-class bias)

{{posthoc_tests}}

Per task (mean over seeds):

{{posthoc_tasks}}

Gains concentrate on CICIoT 8/34-class and Bot-IoT. CICIoMT is mixed: the teacher and the 19-class task gain
(SHIELD 69.24 → 69.71), while the students lose 0.1–0.25 points on binary and 6-class; its test captures differ from
the validation slice, so validation-fitted biases transfer less reliably.

### 4.2 Stronger teacher and models

{{ensemble_teacher}}

Students distilled from this ensemble, compared with the same students distilled from the v2 teacher:

{{ens_students}}

**A stronger teacher does not make better students** here: the 3–15k-parameter students are capacity-limited.

Further detectors (LightGBM, CatBoost on GPU, ResMLP on quantile-transformed inputs with cross-entropy or Balanced
Softmax, and validation-driven ensemble selection, Caruana et al. 2004):

{{detector_v4}}

Model changes move results by about ±1 point; quantile inputs and Balanced Softmax did not help the neural model.
CatBoost is the best single model on CICIoMT 6-class (84.58), but selecting a model by its test score would be
cherry-picking, so the validation-selected ensemble is what we report.

## 5. Deployment

### 5.1 Integer students

v2 used PyTorch dynamic quantisation (one input scale per batch; heavy-tailed features destroy the small ones). v3
uses integer-only inference with a per-feature input scale, per-channel int8 weights and calibrated activations; the
calibration (proportional vs class-balanced rows, clip percentiles) is chosen on validation, and QAT is used when
PTQ loses more than 0.5 points on validation. W8A16 keeps int8 weights with 16-bit activations.

{{int8}}

Rare-class values sit in the feature tails, which is why a single fixed clipping percentile failed in development
and the calibration is selected per model.

### 5.2 Cascade: tiny student, uncertain flows escalated

The student's least-confident flows (max softmax probability; threshold set on validation for each budget) go to an
expert. SHIELD student with the step-2 rule; "matched" = smallest budget whose validation F1 reaches the expert's.

{{cascade}}

On CICIoT 34-class, 10% escalation lifts SHIELD from 65.71 to 70.29 (ensemble alone 71.27). On CICIoMT 6-class,
escalating hurts (82.53 → 81.48 matched), again because of its validation → test shift. Test escalation rates match
the validation budget closely on CICIoT and Bot-IoT; on CICIoMT they exceed it (for example 57.0% at a 50% budget).

### 5.3 Controller coupling with the cascade

Controllers run the student on every flow and the teacher on escalated flows (cost 1/μ_student + e/μ_teacher).
Minimum controllers meeting the relative SLA (CICIoT 34-class capacities):

{{coupling}}

{{coupling_acc}}

The cascade needs **the same number of controllers as the student alone** at every load, while its accuracy
approaches the teacher's (68.13 at 20% vs 68.23). This is a queueing-model result, not a testbed measurement.

## 6. Explanation faithfulness of the SHAP loss

Spearman correlation between the student's and the teacher's SHAP feature rankings, SHIELD (with the attribution
loss) vs kd_shap (same features and size, no loss); 5 seeds × 8 tasks:

{{fidelity}}

SHIELD is more faithful in **33 of 40** pairs (mean +0.119, p = 2.3e-4). Bot-IoT binary goes the other way. Accuracy
is unchanged (v2: SHIELD vs kd_shap, p = 0.10). **Claim: the attribution loss preserves the teacher's explanation
behaviour; it does not improve accuracy.**

## 7. Controller placement

18 methods, each tuned with the same effort on separate tuning graphs and run with exactly 20,000 objective
evaluations (cache hits counted), 30 seeds × 56 instances (8 topologies × k × ρ). Exact optima by parallel exhaustive
search (`brute_force_parallel`, verified identical to the serial search) on 37 instances (up to 10^8 subsets).

{{placement}}

"budget fraction to final" = share of the 20,000 evaluations used before the run is within 0.1% of its final value
(low = early stagnation).

Paired comparisons (method means; 56 instances, and 8 topologies as independent units):

{{placement_pairs}}

Per-instance tests over the 30 seeds (Holm-corrected), original hybrid vs each method:

{{placement_holm}}

Tuning-graph scores (lower is better; ratio to the best result found on each tuning graph):

{{placement_tuning}}

**What the evidence supports:**
- **No single best optimiser.** GWO+SA has the best topology-level rank (2.56) and lowest mean gap (0.304%); the
  diversity-controlled hybrid (`hybrid_linear`) has the best instance-level rank (4.85) and the lowest worst-case gap
  (5.89%). Neither differs significantly from the other (18 vs 15 instances, p = 0.79).
- **Diagnosis of the original EHO–ACO:** premature convergence. It is within 0.1% of its final value after 16.7% of
  the budget; on Cogentco/syn200 the pheromone entropy falls to its MAX-MIN floor and every elephant ends with the
  same k nodes (`analysis/pheromone_diag.py`).
- **Diversity control** (pheromone smoothing + more ant-rebuilt elephants when stagnation and low clan diversity are
  detected; Stützle & Hoos 2000) cuts the mean gap 1.652% → 0.414% and the worst case 22.45% → 5.89%. Per instance
  it is **not uniformly better** (the original is significantly better on 9 instances, worse on 7): it removes the
  catastrophic failures rather than improving typical runs.
- **The fuzzy version of the same controller is worse** than GWO+SA (2 vs 34 instances, p = 2.5e-7) and is not
  adopted. Note: before the proper run, untuned probes of both controllers were made on two evaluation topologies;
  they selected nothing, and all reported numbers come from the tuning-graph → frozen → full-grid protocol.
- **ACO's own contribution is small:** full hybrid vs no-ants is not significant at the aggregate level (16 vs 13
  instances, p = 0.5); per instance over seeds it is significantly better on 4 and worse on 0 (large graphs).
- **A redesigned ACO** (marginal-gain heuristic + MMAS reset) was tried and stopped: worse than the original on the
  tuning graphs (1.0153 vs 1.0078) and on the first completed Kdl instances, and about 20 h for the full grid. Its
  partial runs remain in `runs.jsonl` and are excluded from all tables.
- **Not claimed:** that EHO–ACO (in any version) is the best placement method.

## 8. Attack surge: does detector-driven isolation protect benign switches?

A placement optimised for normal load (ρ = 0.7) faces a surge multiplying the traffic of 10% of switches by M.
Flow costs use the measured cascade escalation rates (benign 15.0%, attack 4.6%; attacks are mostly confident
floods). Strategies are evaluated on the true load; the detector's flow-level precision 99.7%, recall 99.4%,
FPR 6.7% (CICIoT test) decide which switches are flagged. 3 topologies × 20 trials per M.
All network metrics come from the M/M/1 queueing model, not packet-level emulation; "time to mitigate" = model MTTD
(about 0.9 ms) + measured re-optimisation time, without switch-migration time.

{{surge}}

{{surge_tests}}

**Reading:** isolation (one security controller for flagged switches) gives **0% attack exposure** but drops more
benign flows than re-placing with the same number of controllers (k + 1), except at the largest surge (×20: 6.71% vs
6.66%). It beats doing nothing and same-k re-placement on benign drops. This is a trade-off, not a dominance result.

## 9. Attacks never seen in training (leave one family out)

Binary student trained without family F; thresholds at 1% FPR on benign validation flows; 3 seeds.

{{unseen}}

The student alone catches 48.5% of unseen BruteForce and 43.5% of unseen Web attacks (CICIoT). Neither novelty score
(Mahalanobis on the hidden layer, Lee et al. 2018; FlyHash, Dasgupta et al. 2017/2018) improves detection at the
same FPR budget, and FlyHash's FPR does not transfer to CICIoMT's test captures (6.6–6.9% vs a 1% target).

## 10. Limitations

1. CICIoT2023 has no capture identifiers: near-duplicate windows of one capture can sit in train and test (exact
   duplicates are removed). Applies to all published CICIoT2023 results.
2. Literature comparisons use other test sets and mostly a leaky feature; only subset → full is row-matched.
3. Gradient-boosting members and the ensembles use one seed; teacher/student results use 5 seeds.
4. Student settings for the full data reuse the subset's validation tuning (no re-tuning).
5. CICIoMT2024's test captures differ from its validation slice; validation-fitted rules and escalation transfer
   poorly there.
6. Controller coupling and surge results come from a queueing model, not an SDN testbed.
7. Placement statistics depend on the unit: instances (56) are not independent; topology-level tests (8 units) have
   low power.
8. The Nemenyi/Holm results for placement are within one objective function (weights from v2); a weight-sensitivity
   study was not run.

## 11. Files

- Tables: `results/v3/tables/*.csv`; facts used in the text: `results/v3/data/facts.json`.
- Analysis scripts and their outputs: `results/v3/analysis/` (identifiability, learning curve, placement and
  pheromone diagnostics, int8 calibration diagnostics).
- v3 pipeline scripts: `scripts/10_posthoc.py` … `scripts/21_v3_report.py`; new modules `eval/posthoc.py`,
  `eval/cascade.py`, `eval/openset.py`, `eval/predictions.py`, `models/quant.py`, `placement/swarm.py`.
- Raw outputs (not in git, large): `D:\shield_run\outputs_v3`, `D:\shield_run\outputs_full`.
