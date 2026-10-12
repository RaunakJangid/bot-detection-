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
| 7b | Same with row/column subsampling 0.8, 3 seeds (first 3-seed run was deterministic) | test 74.81 / 74.78 / 74.98 (mean 74.86, sd 0.11) vs ens4+rule 73.12: +1.74; validation chose clusters+balanced in 2/3 seeds, hand group in 1 | PASS (rule: gain > 2 sd and > 0.5); caveat: test set seen during development |
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
| 26 | Clipping control (fp32 with the int8 input clipping; CICIoT34, CICIoMT 6/19, 5 seeds) | clipping explains most of int8's advantage under impulse noise (e.g. -1.09 of -1.67 pts) but ~none under Gaussian / missing values, where int8 stays 0.5-1.2 pts more robust than clipped fp32 | PASS (small effect; overlap with Ullah et al. CMC 2026 to discuss) |
| 27 | Realistic evasion (padding x<=10, slowing x<=30, black-box, 20k flows) | overall at max budget: student fp32 2.8%, int8 4.0%, teacher 2.6%, XGBoost 3.1%; BUT rare web classes evade 50-72% (BrowserHijacking, Uploading, CommandInjection) | PASS (overall) / weakness (rare classes) |
| 28 | Capture shift: CICIoMT val (train captures) -> test (new captures) | benign domain AUC 0.75; multiclass models lose 4-13 pts (binary 0.5-2). FPR@1% target -> median 2.5-3.5% across 61-72 models; only 17-22 models stay <= 1.5%; ensembles and XGBoost 6-8%, CatBoost 1.2-2.9% (multiclass). Corrected 2026-10-11 audit: distilled students are NOT significantly better than scratch MLPs (p = 0.35 / 0.55 / 0.068) and one architecture spans 0.3-6.5% across seeds. Inflation tracks each model's F1 drop (Spearman 0.49-0.70) | PASS (finding, reworded) |
| 28b | E1 recalibration after capture change (scripts/36_recalibration.py, cached logits, 61-72 models, 20 draws; rule fixed before run). Primary: burn-in K=2000 flows from new captures, 1% attacks | Fixed val threshold: median FPR 2.5-3.5%, 44-55 models > 1.5%. Clean benign burn-in (conformal quantile): 0.98-0.99%, 0 models > 1.5%, recall loss ~0.1 pt. Label-free contamination-corrected: 1.20 / 1.38 / 1.22%, recall loss <= 0.07 pt, passes for 3-4 of 5 families on 3/3 tasks; FAILS for XGBoost / ensembles (stay 2.6-5.7%) because their false alarms are confident (0.1% target -> 1.6-5.3%). Naive window quantile: FPR ~0 but recall -3.5 to -3.9 pts. CICIoT control: no harm (1.17-1.21% vs 1.08-1.10%). After recalibration all models reach 99.74-99.78% recall at ~1% FPR: the ensembles' fixed-threshold recall edge (99.83%) was bought at 6-8% FPR | PASS (decision rule met; label-free fails on ens/XGB; burn-in is a random sample, no time order) |
| 28c | Far-tail false alarms (0.1% target) by model size, CICIoMT | small MLPs 0.11-0.33% (KD) / 0.11-1.23% (scratch), large neural 0.36-0.79%, ensembles/XGBoost 1.6-4.9%. KD < scratch only on 19-class (p=0.043, n=5 scratch) | PASS (size effect) / KD effect not robust |
| 28d | E2 variance decomposition of FPR transfer (scripts/37_fpr_variance.py; log10 FPR; architectures with >= 3 seeds) | CICIoMT: ICC between architectures 0.12-0.24, i.e. 76-88% of the variance is seed-to-seed (within-arch sd 0.32-0.36 log10 = ~2.2x factor vs between-arch sd 0.23); family differences significant on binary/6-class (Kruskal p 0.02/0.014), not 19-class (0.17); recalibration shrinks within-arch sd 2-3x (to 0.13-0.21). Caveat: multi-seed architectures are mostly small MLPs (GBDT/ensembles single-seed) | PASS (seed dominates) |
| 28e | E7 real benign traffic in the CICIoT2023 feature space (scripts/40_real_benign.py; SentryIoT extractor, MIT; UNSW smart-home traces 3 days + IoT-23 honeypot pcaps) | INVALID as a false-alarm measurement: the Kaggle 46-column CICIoT2023 rows are not reproducible from pcaps with the authors' published extractor. 6 shared-name columns have different definitions (Header_Length median 5e5 vs ~20; ack/rst_count, Number 5.5, Variance 0.9 while Std ~300) and rows are averages over several windows (Protocol Type 6.5). Domain AUC lab-benign vs real-benign = 1.00 with 37 or 31 columns, so any FPR (XGBoost 72-100%, MLPs 2-82%) measures representation mismatch | FAIL (infeasible in CIC space) / dataset finding |
| 28f | E3 second capture-shift dataset: IoT-23 (scripts/41_iot23_shift.py; Zeek flow features, no addresses/ports; 7 of 23 captures held out incl. Somfy honeypot; strict de-dup; 16 models) | Shift REPLICATES and is stronger: benign domain AUC 0.97 (control 0.50); FPR at 1% validation target 8.5-35% (control 1.03-1.30%); trees best here (XGB 8.5, RF 10.1), MLPs worst (19-35%) - family ranking is dataset-dependent. Held-out captures = unseen malware families: recall collapses to 2-6%, test macro-F1 15-27 vs val 97-98 | PASS (replicates; decision rule met) |
| 28g | E1 with TIME-ORDERED burn-in (IoT-23, first 1000 benign flows in time per test capture; 3 captures with enough benign) | clean burn-in: median FPR 1.35% (XGB) to 2.5-4.3% (MLPs), capture 35-1 stays 2-3% after 20 h; label-free corrected 0.9-4.4% with recall collapse for small MLPs. Random-sample burn-in (28b) overstates: within-capture temporal drift breaks exchangeability | FAIL for "budget restored" in time order; superseded by 28j |
| 28f2 | E3 v2 (rerun 2026-10-11): XGB/RF now 5 seeds (subsample 0.8); designs: random capture hold-out (mixed: 35-1/52-1 are Mirai, a training family; 1-1, 8-1, 9-1, 60-1 unseen), SAME-FAMILY hold-out (17-1 Kenjiro, 20-1 Torii, 49-1 + 7-1 Mirai; all families in training) and random control | Same-family (benign drift only): domain AUC 0.993; FPR at 1% target 9.0-38.7% (XGB 36-39, RF 36-37, large MLP 26-33, small MLP 9-27); random hold-out 7.5-37%; control 0.98-1.35% | PASS: benign drift alone breaks the budget |
| 28j | E8 controlled streaming comparison (scripts/42_e8_stream.py; IoT-23 time order; fixed / random burn-in / time burn-in / rolling label-free / delayed-audit / Nguyen et al. 2026 re-implementation; rule fixed before run: median pooled FPR <= 1.5% and recall loss <= 5 pts) | Rule FAILS for every time-ordered method. Same-family (K=1000, 1 capture): fixed 11.7%, burn-in-time 0.9% but recall -16.7, rolling free 3.1% (-3.7), audit 9.0%, Nguyen 14.7%. K=500 (2 captures): fixed 35.0%, burn-in-time 2.2% (-34.4 recall), rolling 10.2%, Nguyen 39.1%. Random-capture hold-out: only random-sample burn-in passes (0.66-0.75%), time burn-in 3.2-4.4%, rolling 1.9-2.7%, recall ~0-21% throughout. Nguyen gate worse than fixed on every model, ~10 us/flow | FAIL (method); finding: time order matters; no threshold method rescues lost ranking |
| 28k | Discrimination under shift (results/v3/analysis/shift_discrimination.py; ROC/PR-AUC val vs new-capture test) | CICIoMT: ROC-AUC 0.9998 -> 0.9995 median (61-72 models): ranking intact, only the threshold moves. IoT-23: same-family 0.997 -> 0.669; random hold-out 0.992 -> 0.064 (inverted orientation); control 0.997 -> 0.996 | PASS: two failure regimes (threshold shift vs ranking loss) |
| 28l | E4 main effects and interactions (results/v3/analysis/protocol_interactions.py; 2^k contrasts, single run per cell) | IAT +7.6 to +15.2 macro-F1 (+10.5 to +18.1 acc); random split +4.1 to +10.5 F1 (CICIoMT); de-dup -2.0 to +0.8; IAT x split -1.7 to -2.7 (both leak capture identity) | PASS (descriptive) |
| 28m | CICIoT2023 window-size artefact (scripts/44_window_shortcut.py; dataset paper Sec. 4: 10- vs 100-packet windows) | Number/Weight take 3 value pairs by class group (5.5/38.5, 9.5/141.55, 13.5/244.6); alone: binary 85.4 F1 / 97.2% acc, family 19.1 / 67.0%; removing them: -0.07 (binary), -0.23 (family) F1 | PASS (artefact exists, models do not depend on it) |
| 28o | Commissioning screen: local benign memory (kNN / 64 centroids) fused with detector, IoT-23 time order (scripts/45_commission_screen.py; rule fixed before run) | Same-family K=500 (17-1, 49-1): fixed 35.0% FPR / 96.2% recall; burn-in-time 2.2% / 63.2%; memory+detector 1.6-1.9% / 88.2-88.3% (ROC 0.96-0.98). K=1000: 0.9% / 83.4% (loss 12.8). Random hold-out: within-capture base ROC 0.48 -> fused 0.75, recall ~21%. CORRECTION: within-capture ROC on same-family is 0.96, so its pooled 0.67 was a cross-capture artefact | FAIL by rule (near miss, +25 recall vs best E8 method at similar FPR) |
| 28p | Commissioning idea, leave-one-capture-out (scripts/46_commission_loco.py; config frozen on tuning captures) | Only 4 eligible captures (2 tuning, 2 evaluation). Frozen kNN/K1000/w0.7: 49-1 (ranking intact) burn-in alone 0.59%/99.5% = memory 0.47%/99.5%; 3-1 (ranking lost, ROC 0.51->0.70) burn-in 2.7%/14.4% vs memory 5.0%/33.8%. Contaminated commissioning window: recall 49.8% (1% attacks), 10.6% (5%), 4.2% (10%) | FAIL: no gain where ranking is intact, too weak where lost, collapses with realistic contamination; idea dropped |
| 28q | Multi-student: averaging 5 seeds of the same 16 KB student (validation) | +0.4 to +0.7 macro-F1 for 5x size | FAIL (not worth it) |
| 28r | E9 tiny rare-class expert (scripts/47_tiny_expert.py; validation half 2) | student 66.8 F1 / 4.5% benign false alarms; + expert (CE) 68.0 / 9.5%; + expert (KD) 67.5 / 8.1%; same-recipe single 2x 62.1 / 12.0% (recipe weaker than SHIELD: 1x 63.8) | FAIL (rule: >= +1.5 and no benign-FA increase) |
| 28t | E5 complete (holdout, 2M unseen rows) | Experts (subsample 0.8, 3 seeds): 75.41 / 75.39 / 75.34 vs ens4+rule 73.79 -> +1.59 (old test +1.74). Holdout student/cascade INVALID: the fullho student fell back to default KD settings (k 16, 4,290 params) instead of the full-data settings (k 32, 14,690 params); validation 55.9 vs 66.6 on identical rows | PASS (ensemble, experts); student/cascade holdout to be rerun with the full-data settings |
| 28s | CRITERIA CORRECTION (2026-10-12): recall loss vs the fixed threshold is invalid (it fires on 11-74% of benign); threshold-only methods sit on the detector's own ROC, so they are judged on hitting the FPR target; score-changing methods vs the detector oracle at the same FPR; E9 at matched benign false alarms (results/v3/analysis/e8_oracle.py) | E8 corrected: time-ordered burn-in holds the budget on same-family captures (0.88% K1000; 1.56% K500), fails on the random hold-out (2.9-3.6%); rolling and Nguyen still fail. Memory (LOCO, 4 captures): +11 recall over detector oracle on 3-1 (ranking lost, at 5% FPR), 0 on 49-1; contaminated window still collapses, detector-filtered window does not fix it (-25 to -47 vs oracle). E9 at matched false alarms: 67.5 F1 / 3.8-3.9% vs student 66.8 / 4.5% (+0.7) | Memory FAIL (contamination); E9 small real gain (+0.7, below +1.5); burn-in claim partly restored |
| 28n | Tree/ensemble seed replicates on CICIoMT (scripts/43_tree_seeds.py: xgbsub s0-4, lgbm/catboost s1-2, treeens s0-2) then rerun of shift/recalib/variance (73-84 models) | Trees stable across seeds and consistently high: XGB-sub 8.0-8.9% (binary), 6.9-7.6% (6-cl), 5.4-6.9% (19-cl); tree ensemble 5.7-7.7%; LightGBM/CatBoost task-dependent (0.7-7.3%). Small MLPs 0.27-6.5% across seeds. ICC (architecture share) now 0.26-0.41 at 1%, 0.30-0.56 at 0.1% (was 0.12-0.24 with MLP-only seeds). Headline unchanged: fixed median 2.8-3.9%, clean recal 0.98-0.99%, label-free 1.22-1.38% | PASS; E2 claim revised to "comparable, family-dependent" |
| 28h | E4 protocol-inflation factorial (scripts/39_protocol_ablation.py; XGBoost + RF, 2M-row cap, same settings in every cell) | Strict -> published-style: CICIoT34 XGB 68.3/88.0% (F1/acc) -> IAT kept 75.8/98.9% -> +no de-dup 76.7/99.3%; RF 66.3/86.1% -> 80.6/99.4%. CICIoMT 19-class XGB 71.2/80.1% -> random split 81.9/86.1% -> + IAT 93.6/99.7% (+22 F1, +20 acc). IAT is the largest factor (+7 to +15 F1, accuracy to ~99%); random split second (+6 to +11 F1 on CICIoMT); de-dup small and mixed (-5 to +2) | PASS (descriptive) |
| 28i | E5 fresh holdout (scripts/38_make_holdout.py + chain_holdout.ps1; 2M rows carved from the full train pool, scaler refit, pipeline retrained, holdout scored once) | ens4+rule: holdout 73.59 vs old test 73.22 (val 74.21); ens4 73.92 vs 72.63; teacher 70.30 (5-seed mean); XGBoost 71.37 vs 70.81; TabM 71.01. The reused test set did NOT inflate the estimate. Experts step crashed (CUDA OOM, GPU shared with other jobs); resumes via chain_holdout2.ps1 after the GPU frees | PASS (base); experts PEND |
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
- Almahaqeri et al., Sci. Rep. 2026 (XGBoost, CICIoT2023 2/8/34-class): 34-class macro-F1 88.8, 8-class 89.0,
  binary 99.5. Protocol: random 80/20 split of an undersampled 4-5M set, NO cross-split de-dup, IAT KEPT (IAT is
  among their top features). Their minority-class F1 (Uploading 0.62, SqlInjection 0.64, XSS 0.65, PingSweep 0.66)
  shows the same weak classes as ours. Their listed future work (cross-dataset, natural prevalence, drift,
  adversarial evasion) is what our items 27-32 already do.
- Rodriguez et al., Internet of Things (Elsevier) 2026, rule-based label harmonization (Zeek + MITRE ATT&CK/CAPEC):
  within-dataset F1 > 0.99, cross-dataset (train CIC-IoT-2023 -> IoT-23/IoTID20) avg F1 < 0.12 for trees, 0.99
  only for MLP; fusion restores > 0.99 (random 40/60 split, no de-dup). Agrees with our item 32 (cross-dataset
  collapse) and our observation that neural students transfer better than trees. Uses accuracy-style averages and
  random splits, so its in-domain numbers are optimistic; no seeds/CIs.
- Low-tier papers reviewed (WSN SSA-GWO, GWO-SVM Bot-IoT, Pokhrel KNN): leakage (SMOTE before split) or synthetic data; not benchmarks.
- Top-venue evaluation axes covered: leakage-free protocol, per-class metrics, CIs, operating point, base rate, capture shift with control, cross-dataset, unseen attacks, constrained adversarial evasion, edge cost.

## 6. Pending

- 43 Mininet. Next proposed: E2-E7 in research/audit/EXPERIMENT_DECISION_PLAN.md (E1 done, 28b).
- Audit 2026-10-11: see `research/audit/` (FINAL_RESEARCH_AUDIT.md, EXPERIMENT_DECISION_PLAN.md E1-E7).
- Paper writing (P1 detection, P2 placement): later.
