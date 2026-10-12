# Claims and limitations (what P1 may and may not say)

Status key: **CLAIM** = supported, may be stated; **QUALIFY** = state only with the given wording;
**DROP** = do not claim; **PEND** = wait for the named run.

## Detection

| Statement | Status | Wording / reason |
|---|---|---|
| Full CICIoT2023 data improves the same model on identical test rows (70.98 → 73.22) | CLAIM | Row-matched; one GBDT seed |
| Experts add +1.7 macro-F1 (74.86 ± 0.11 vs 73.12, 3 subsampled seeds) | CLAIM (after E5 holdout) | Test set reused during development; confirm on fresh holdout |
| Validation-fitted decision rule helps | QUALIFY | "small gain (+0.7–1.0) on CICIoT and Bot-IoT; mixed on CICIoMT"; re-test with tasks as units |
| We beat published results | DROP | Different test sets; published numbers mostly keep IAT. Say "under a strict protocol, published pipelines score X" |
| IAT is label leakage | QUALIFY | "IAT acts as a collection-specific shortcut: two-stage CatBoost 78.1 → 92.1 when kept". Supporting evidence: the dataset paper's own Table 5 reports IAT mean 8.32e7 with sd 1.70e7, inconsistent with an inter-packet time and consistent with a capture-time value |
| Protocol choices inflate scores (E4) | CLAIM, exact wording only | "In the evaluated CICIoT2023 and CICIoMT2024 configurations, protocol choices materially changed the apparent performance of the same XGBoost pipeline. CICIoT2023 34-class: 68.3% macro-F1 / 88.0% accuracy strict -> 76.7% / 99.3% with IAT retained and cross-partition de-duplication omitted. CICIoMT2024 19-class: 71.2% / 80.1% strict (official capture split) -> 81.9% / 86.1% with a random row split -> 93.6% / 99.7% with a random split and IAT retained (86.0% / 98.8% with IAT retained under the official split). These results quantify sensitivity to the specified choices; they do not establish that published high scores are invalid." Main effects (2^k contrasts, single run per cell): IAT +7.6 to +15.2 F1; random split +4.1 to +10.5 F1 (CICIoMT); de-dup -2.0 to +0.8 F1; IAT x split -1.7 to -2.7 (partial substitution). No capture-level split claimed for CICIoT2023 |
| "Published 99% results are protocol artefacts" | **DROP** | Overbroad; see wording above |
| CICIoT2023 window-count features are collection artefacts | CLAIM (pending family/34-class arms) | Number/Weight take three value pairs (5.5/38.5, 9.5/141.55, 13.5/244.6) tied to the 10/100-packet windows of the dataset paper (Sec. 4); Number+Weight alone give 85.4 macro-F1 / 97.2% accuracy on binary; removing them costs 0.07 F1 (redundant with other features) |
| Remaining headroom is small | CLAIM | 1-NN Cover–Hart bound 9.0–17.2% error vs XGBoost 11.8% (CICIoT 34) — asymptotic estimate |

## Operating point and shift

| Statement | Status | Wording / reason |
|---|---|---|
| Under capture shift a 1% validation FPR becomes 2.5–3.5% (median), 6–8% for ensembles/XGBoost | CLAIM | 61–72 models; control AUC 0.50 with no inflation |
| Distilled students hold the FPR budget | **DROP** | KD vs scratch MLP p = 0.35 / 0.55 / 0.068; seed range 0.27–6.5% |
| Trees break, neural nets hold | **DROP** | CatBoost 1.2–2.9% on multiclass; TabM/teacher 4.6–6.2% |
| FPR inflation tracks the model's own accuracy drop | CLAIM | Spearman 0.49–0.70 |
| Seed variance dominates architecture for FPR transfer | **QUALIFY** (revised after tree/ensemble seeds, 2026-10-11) | With 16–18 multi-seed architectures incl. XGBoost/LightGBM/CatBoost/tree-ensemble seeds: architecture explains 26–41% of log-FPR variance at the 1% target (ICC), seed 59–74%; at 0.1% architecture 30–56%. Say: "seed-to-seed variation is comparable to differences between architectures, and depends on the family: XGBoost and tree ensembles inflate the budget consistently and stably across seeds (binary 7.5–8.9%), small MLPs are lower on median (1.7–2.9%) but unstable across seeds (0.27–6.5%)" |
| Recall ~99% at 1% FPR on CICIoT | CLAIM | In-distribution only |
| Benign drift alone breaks the budget (no unseen attacks needed) | CLAIM | IoT-23 same-family split (Kenjiro, Torii, Mirai x2 held out; families seen in training): 1% target -> 9–39% FPR; random-split control 0.98–1.35% |
| Two failure regimes: threshold shift vs ranking loss | QUALIFY (corrected 2026-10-12) | CICIoMT: ranking intact (ROC 0.9995), threshold moved. IoT-23 same-family: pooled ROC 0.67 is a cross-capture artefact; WITHIN each capture ROC is 0.96 -> per-capture threshold shift (fixed FPR up to 35–50%). Genuine ranking loss only on the random capture hold-out (within-capture ROC 0.48; pooled 0.06). Report within-capture metrics, not pooled |
| Local benign memory repairs per-capture threshold shift | PEND (screen near-miss) | 45_commission_screen.py, same captures, K=500: memory+detector FPR 1.6–1.9% at 88% recall vs time-ordered burn-in 2.2% at 63% recall (+25 pts); fails the pre-set rule (FPR <= 1.5%, loss <= 5) by a small margin; only 2 captures; needs leave-one-capture-out test |
| Recalibration restores the budget | QUALIFY | Only shown in the intact-ranking regime with a random-sample window (CICIoMT 0.98–0.99% at ~0.1 pt recall). CICIoMT has no timestamps, so time-ordered recalibration in that regime is untested. On IoT-23 (ranking-loss regime), E8 decision rule FAILS for every time-ordered method: burn-in 0.9–2.2% median FPR but recall loss 17–34 pts (same-family); rolling label-free 3.1% (loss 3.7, K=1000) / 10.2% (K=500, two captures); delayed-audit 9–33%; random-capture hold-out: burn-in 3.2–4.4%, rolling 1.9–2.7% with recall near 0. Random-sample windows look better (0.6–0.9%) than any time-ordered window on the same captures |
| Time-ordered recalibration (corrected criterion, 2026-10-12) | QUALIFY | Judged on hitting the FPR target (threshold-only methods cannot exceed the detector's ROC): time-ordered burn-in holds it on IoT-23 same-family captures (0.88% K1000, 1.56% K500; 1-2 captures), fails on the random hold-out (2.9-3.6%). The earlier "fails everywhere" used an invalid recall-loss-vs-fixed criterion |
| Experts gain holds on unseen data | CLAIM | E5 holdout: +1.59 (3 seeds, 75.34-75.41 vs 73.79) |
| Rolling recalibration is a method contribution | **DROP** (unless a dataset with intact ranking and timestamps shows otherwise) | E8 on IoT-23; the regime diagnosis explains why |
| Nguyen et al. feedback-guided adaptation holds the budget on real captures | DROP (our re-implementation) | Worse than the fixed threshold on every model (gate adds false alarms); state as re-implementation from published equations with unnamed gate features chosen on validation |
| A random-sample calibration window represents deployment | DROP | Within-capture time order matters (IoT-23) |
| A benign burn-in of ~2000 flows restores the 1% budget for all 61–72 models (0.98–0.99%) at ~0.1 pt recall cost | CLAIM | Burn-in is a random sample of new captures; valid by exchangeability; no time order |
| Label-free contamination-corrected quantile restores it to 1.2–1.4% median | QUALIFY | Works for small/large neural and most GBDT; fails for XGBoost/ensembles (confident false alarms) |
| Ensembles' recall edge under shift is bought with false alarms | CLAIM | 99.83% at 6–8% FPR vs 99.74–99.78% for all models at ~1% after recalibration (binary detection) |
| Large ensembles produce confident false alarms under shift; small models' are marginal | CLAIM | 0.1%-target FPR: 1.6–5.3% vs 0.11–0.47% |

## Edge model

| Statement | Status | Wording / reason |
|---|---|---|
| Integer-only int8 student loses ~1 point at 3.7–16.4 KB | CLAIM | vs PyTorch dynamic quantisation 14.9 (old baseline, say so) |
| int8 is more robust to noise than fp32 | QUALIFY | "0.5–1.2 pts smaller drop under Gaussian/missing-value noise, beyond input clipping; under impulse noise mostly explained by clipping". Small effect; Ullah et al. 2026 overlap |
| Entropy cascade reaches ensemble accuracy at ~10% escalation | CLAIM | Full data 73.51 vs 73.21; 5 seeds, sd < 0.1 |
| Learned escalation gate | DROP | Below entropy at every budget over 5 seeds |
| 0.023 ms/flow | QUALIFY | "one desktop CPU core"; not a device measurement |
| Cascade needs the controllers of the student alone | QUALIFY | M/M/1 model, not a testbed |

## Robustness and explainability

| Statement | Status | Wording / reason |
|---|---|---|
| Constrained evasion succeeds on 2.6–4.0% of detected attack flows | CLAIM | Black-box padding/slowing grid, 20k flows |
| Models are robust to evasion | **DROP** | Rare web classes evade 50–72% |
| Novelty detectors (Mahalanobis, FlyHash) improve unseen-attack detection | DROP | Negative result; report it |
| SHAP loss gives more faithful explanations | QUALIFY | Re-test with tasks as units; minor result; SHAP+KD exists |
| SHAP-guided KD is our contribution | DROP | Null accuracy effect; prior art (arXiv 2512.19488) |

## Limitations to state in the paper

1. CICIoT2023 CSVs have no capture IDs; near-duplicates within a capture can cross splits (applies to all published results).
2. Capture shift is shown on two datasets (CICIoMT2024, IoT-23); IoT-23 has few test captures with enough benign flows for time-ordered recalibration (1–2 at K = 1000; sensitivity run at K = 500).
3. Test sets were inspected during development; the fresh holdout (E5) gives 73.59 vs 73.22 on the old test for the base ensemble.
4. Tree/ensemble seed replicates on CICIoMT added late (43_tree_seeds.py); IoT-23 trees have 5 seeds.
4b. CICIoT2023 real-traffic evaluation (E7) is infeasible in the Kaggle 46-feature space: per the dataset paper (Table 4) several features are flow-level ratios/products averaged over 10/100-packet windows (e.g. Variance = ratio of incoming to outgoing variance), and the available 39-column extractor does not compute them; lab vs real benign windows are perfectly separable (AUC 1.0). Real-home traffic is evaluated in the Zeek space instead.
5. Network results are queueing-model based until the Mininet/Ryu run.
6. No IoT hardware latency or energy measurement.
7. Placement (EHO–ACO) is out of scope for P1; GWO+SA is published prior art.
