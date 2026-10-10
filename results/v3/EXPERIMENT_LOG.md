# SHIELD-IoT experiment log (living document)

Last updated: 2026-10-11. All numbers are under the strict protocol unless marked otherwise:
IAT (CIC) and seq (Bot-IoT) dropped, exact duplicates removed across train/validation/test, scaler fitted on
train only, every choice (thresholds, weights, groups, configurations) made on validation, test scored once.
Main metric: macro-F1 (%). "Full" = CICIoT2023 full Kaggle data (25.8M train rows), same validation/test rows as
the strict subset.

Legend: PASS = kept / positive result, FAIL = no gain or worse (dropped), PEND = queued or not run.

## 1. Detection accuracy

| # | Experiment | Result | Verdict |
|---|---|---|---|
| 1 | Validation-fitted decision rule (logit adjustment + per-class bias) | teacher +0.98 (35/40 tasks, p=1.6e-9); SHIELD student +0.67 | PASS |
| 2 | Ensemble teacher (5 ResMLP + XGBoost, weight on validation) | best on 7/8 tasks | PASS |
| 3 | Students distilled from the ensemble teacher | not better than from the single teacher | FAIL |
| 4 | Caruana ensemble ens4 (LightGBM, CatBoost, quantile ResMLPs, teacher, XGBoost) + rule | subset 70.98; full 73.22 [72.46, 73.90] | PASS |
| 5 | Full data (6.4x training rows) | +2.4 over XGBoost on identical test rows | PASS |
| 6 | Low-volume specialist expert (hand group) | 74.71 [73.93, 75.41], +1.49, p<0.002 | PASS |
| 7 | Experts per confusion cluster + fully balanced rare-class weights (chosen on validation) | 74.74, +1.62 over ens4+rule (73.12) | PASS (seed spread PEND, see 7b) |
| 7b | Same with row/column subsampling 0.8, 3 seeds (first 3-seed run was deterministic) | queued (chain_sub.ps1) | PEND |
| 8 | Label cleaning on train (2-fold cross-fitted) | no gain | FAIL |
| 9 | Two-level classifier (family -> class) | validation chose the flat model (73.01) | FAIL |
| 10 | Stacking (meta-learner on member log-probs) | 64.89 | FAIL |
| 11 | Learned escalation gate (5 seeds) | unstable, below entropy/confidence | FAIL |
| 12 | TabM (ICLR 2025) as a baseline | best deep model: 70.12 (CICIoT34), 80.97 / 71.12 (CICIoMT 6/19) | PASS (baseline) |
| 13 | TabM added to the ensemble | 73.12 vs 73.22 | FAIL |
| 14 | Deep baselines CNN-LSTM / CNN-GRU / FT-Transformer | 67.49 / 67.92 / 68.71 (CICIoT34) | baseline |
| 15 | Metaheuristics tuning thresholds / ensemble weights (EHO-ACO, GWO+SA, PSO, SA, coord, random; equal budget, 5 seeds) | all tie (p >= 0.19) | FAIL (saturated problem) |
| 16 | Binary CICIoT | 95.70 (dataset paper 96.5 without de-dup) | weak |
| 17 | CICIoMT 6-class / 19-class (ours) | 83.93 / 71.6; published two-stage CatBoost under our protocol 78.13 / 62.01 (92.12 with IAT) | PASS |

## 2. Edge model (student, int8, cascade)

| # | Experiment | Result | Verdict |
|---|---|---|---|
| 18 | Integer-only int8 student (calibration grid on validation, QAT if needed) | mean drop 1.05 pts (v2: 14.9); 3.7-16.4 KB | PASS |
| 19 | Cascade student -> ensemble | 65.71 -> 70.29 at 10% escalation; controllers 3 vs 7 (Geant, rho 0.9) | PASS |
| 20 | Entropy instead of confidence as escalation score (5 seeds) | +0.2 to +0.4 at every budget | PASS |
| 21 | TabM student vs parameter-matched MLP student (3 seeds) | 66.90 vs 66.91 (CICIoT34); tie on CICIoMT | FAIL (tie) |
| 22 | Tiny LightGBM student at equal size (<= 17 KB) | 61.7 / 79.0 / 67.0 vs MLP 66.9 / 80.7 / 70.6 | MLP wins (PASS for our choice) |
| 23 | Edge profile, 1 desktop core | 0.023 ms/flow, 2.4M flows/s, 16 KB int8 (not a Raspberry Pi) | PASS (caveat) |
| 24 | Energy / real device | no hardware | not possible |

## 3. Robustness, shift, explainability

| # | Experiment | Result | Verdict |
|---|---|---|---|
| 25 | int8 vs fp32 under Gaussian / impulse / missing-value noise (5 seeds, 4 tasks) | int8 drops LESS than fp32 almost everywhere | PASS (clipping control PEND) |
| 26 | Clipping control (fp32 with the int8 input clipping) | queued (chain_eval.ps1) | PEND |
| 27 | Realistic evasion (padding x<=10, slowing x<=30, black-box, 20k flows) | evasion ~1-3% for student fp32/int8 and teacher; XGBoost running | PASS |
| 28 | Capture shift: CICIoMT val (train captures) -> test (new captures) | benign domain AUC 0.75; all models lose 5-13 pts; FPR@1% -> 3-8% for trees/ensemble, 0.5-3% for distilled students | PASS (finding) |
| 29 | Control: CICIoT random split | AUC 0.50, no gap, FPR@1% holds | PASS (control) |
| 30 | Recall at fixed FPR (threshold on validation) | CICIoT ~99.1% recall at 1.0% FPR | PASS |
| 31 | ROC-AUC / PR-AUC | ROC ~99.7 for all; PR-AUC separates (CICIoT34 specialist 78.3; CICIoMT student best 89.5 / 74.8) | PASS |
| 32 | Cross-dataset CICIoT <-> CICIoMT (v1/v2, shared 5 classes) | large drop (shared5 ~77-79 -> ~40-48 zero-shot); few-shot recovers | done (finding) |
| 33 | Unseen attacks (leave-one-family-out) | student ~45% at 1% FPR; Mahalanobis / FlyHash novelty does not help | FAIL (novelty) |
| 34 | SHAP fidelity SHIELD vs KD-SHAP | 33/40 pairs, p=2e-4 | PASS |
| 35 | SHAP-alignment loss (v1/v2) | null | FAIL |

## 4. Placement / network (P2)

| # | Experiment | Result | Verdict |
|---|---|---|---|
| 36 | 12 metaheuristics, tuned on separate graphs, vs exact optimum | GWO+SA best (gap 0.30%, within 1% in 95.9% of runs, 1.4 s/run); GWO+SA is PUBLISHED (hGWO-SA), not ours | baseline |
| 37 | Original EHO-ACO | gap 1.65%, worst 22.4%, premature convergence | FAIL |
| 38 | EHO-ACO + linear stagnation/diversity control | gap 0.41%, tie with GWO+SA (p=0.79) | tie |
| 39 | EHO-ACO + fuzzy control | worse than GWO+SA (p=2.5e-7) | FAIL |
| 40 | EHO-ACO + learned-consequent ("credit") fuzzy | 0/8 topologies vs GWO+SA; idea exists in hyper-heuristics | FAIL |
| 41 | ACO2 redesign (marginal heuristic) | far too slow | FAIL |
| 42 | Attack surge (M/M/1 model) | isolation: 0% exposure; benign drop = best re-placement at x20 (6.7%) | PASS (modelled) |
| 43 | Mininet + Ryu emulation | after reboot (Monday) | PEND |

## 5. Literature checks

- Strong prior work on our datasets uses IAT and/or no de-dup (e.g. Abid et al. 2025, GWO selected IAT on both datasets); their higher numbers are not comparable.
- Low-tier papers reviewed (WSN SSA-GWO, GWO-SVM Bot-IoT, Pokhrel KNN): leakage (SMOTE before split) or synthetic data; not benchmarks.
- Top-venue evaluation axes covered: leakage-free protocol, per-class metrics, CIs, operating point, base rate, capture shift with control, cross-dataset, unseen attacks, constrained adversarial evasion, edge cost.

## 6. Pending

- 7b experts with subsampling (seed spread), 26 clipping control, 27 XGBoost evasion, 43 Mininet.
- Paper writing (P1 detection, P2 placement): later.
