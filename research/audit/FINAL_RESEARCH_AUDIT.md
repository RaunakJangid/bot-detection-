# SHIELD-IoT: final research audit

Date: 2026-10-11. Scope: every claim in `results/v3/EXPERIMENT_LOG.md` and `results/v3/FINAL_REPORT_v3.md`,
checked against the raw outputs in `D:\shield_run\outputs_v3` and `outputs_full`, plus a literature check.
Method: read-only scripts (kept in the session scratchpad) that recompute the headline numbers from the output JSON.
Companion files: `CLAIMS_AND_LIMITATIONS.md`, `RELATED_WORK_MATRIX.md`, `EXPERIMENT_DECISION_PLAN.md`,
`REPRODUCIBILITY_AUDIT.md`, `BIBLIOGRAPHY.bib`.

## 1. Verdict in one paragraph

The engineering results are real and reproduce from the output files: full-data ensemble 73.22, experts 74.74,
int8 loss about 1 point, the entropy cascade matching the ensemble at about 10% escalation. None of them is
novel enough to carry an IEEE journal paper by itself. The strongest material is the **operating-point
analysis under capture shift** (CICIoMT2024, 61–72 models per task, with a random-split control on CICIoT2023).
The audit found that the log's wording of that result is **wrong in one important way**. It is not true that
distilled students hold the false-alarm budget. Almost every model breaks it, and the spread across seeds of one
architecture is as large as the spread between model families. This changes the paper's thesis (section 4).
Recent work already covers target-FPR transfer and conformal alerting under shift (Jin & Lee 2026; Ha & Kim 2026;
CALIBURN; FADES). Any calibration contribution must therefore be positioned against them, not claimed as new.

## 2. Claims verified against raw outputs

| Log # | Claim | Recomputed | Status |
|---|---|---|---|
| 4 | ens4+rule full CICIoT 34-class 73.22 | 73.22 (specialist JSON `test_without`) | OK |
| 6 | Specialist 74.71, +1.49 | 74.71 vs 73.22 | OK |
| 7 | Experts clusters+balanced 74.74, +1.62 over 73.12 | 74.74 vs 73.12; chosen config identical in all 3 seeds and test metrics bit-identical | OK, seed spread still unmeasured (7b running) |
| 11 | Learned gate unstable, below entropy/confidence | 5 seeds at 10%: gate 72.61 ± 1.18, confidence 72.99 ± 0.14, entropy 73.31 ± 0.03, entropy+benign 73.51 ± 0.08 | OK. **But** `gate/ciciot_full_class34.json` holds seed 0 only, where the gate looks best; it must not be cited |
| 19/20 | Cascade; entropy better than confidence | Full data: entropy+benign reaches 73.51 at 10% vs ensemble alone 73.21; confidence 72.99 | OK (and stronger than the log says) |
| 25 | int8 drops less than fp32 under noise | int8 smaller drop in 148/160 seed × corruption cells; margins 0.2–5 points; Bot-IoT mild Gaussian is the exception | OK as stated; effect small; clipping control pending (the obvious confound) |
| 27 | Evasion ~1–3% | Overall at S≤10, T≤30: fp32 2.8%, **int8 4.0%**, teacher 2.6%, XGBoost 3.1%. Per class: BrowserHijacking 72% (fp32), Uploading 72% (int8), CommandInjection 51% (teacher) | **Wording fix needed**: "2.6–4.0% overall, but 50–72% for the rare web classes" |
| 28 | Capture shift: FPR@1% → 3–8% trees/ensembles, 0.5–3% distilled students | See section 3 | **Incorrect as worded** |
| 29 | CICIoT random-split control: no gap | Domain AUC 0.500 / 0.503 (full); FPR@1% 0.98–1.27% for every model except LightGBM (0.01%, degenerate thresholds) | OK, note LightGBM anomaly |
| 17 | Two-stage CatBoost 78.13 strict / 92.12 with IAT | Matches report tables | OK |

## 3. The capture-shift result, re-analysed

Source: `outputs_v3/shift/shift_base_rate.json` (threshold set for 1% FPR on validation benign flows, applied to
the official CICIoMT2024 test captures). Benign domain classifier AUC 0.748 (CICIoMT) vs 0.500 (CICIoT control).

| CICIoMT task | models | median test FPR at 1% target | models within 1.5% | XGBoost | ensemble / ens4 | CatBoost | distilled students (median, n=40) | scratch MLP (median, n=5) | KD vs scratch (Mann–Whitney p) |
|---|---|---|---|---|---|---|---|---|---|
| binary | 61 | 2.65% | 17 | 8.30% | 8.30 / 7.56% | 6.67% | 2.29% | 2.98% | 0.35 |
| 6-class | 72 | 3.54% | 17 | 6.98% | 6.54 / 5.77% | 2.91% | 2.33% | 3.22% | 0.55 |
| 19-class | 72 | 2.48% | 22 | 6.96% | 6.38 / 6.56% | 1.20% | 2.23% | 3.75% | 0.068 |

- **Single-seed spread is huge.** SHIELD student test FPR across 5 seeds: 0.27–4.83% (6-class), 0.64–6.51%
  (19-class). Scratch MLP: 0.91–5.45%. Whether a model "holds" the budget is not a property of its architecture.
- **What predicts inflation:** the model's own val→test macro-F1 drop (Spearman 0.65 / 0.70 / 0.49, p ≤ 1e-5).
  Validation accuracy barely does (0.17 / 0.33 / 0.14).
- **Consistent pattern:** validation-selected ensembles and XGBoost are among the worst on every task (6–8%).
  CatBoost is not; "trees break, neural holds" is therefore also wrong.
- **What can be claimed:** under real capture shift, a validation-set FPR budget is not a reliable operating point
  for any model family; the most accurate (ensemble) models inflate it most; the control shows the effect is
  caused by the capture change, not the threshold procedure. Validation and test logits are cached for every
  model (`outputs_v3/logits`), so recalibration methods can be evaluated offline.

## 4. Revised thesis for paper P1

Old (2026-10-11 plan): "distilled edge students hold the FPR budget; label-free recalibration as method".
Revised after audit:

> Under capture shift, every family of IoT intrusion detector loses its false-alarm budget (median 2.5–3.5×,
> ensembles 6–8×), and seed-to-seed variation dominates architecture. We show this with 72 models under a
> leakage-free protocol and a random-split control, and evaluate how much of the budget can be recovered without
> attack labels at an integer-only edge model with selective escalation.

Novelty margin: the shift-and-alert-budget theme is now occupied (Jin & Lee 2026, Electronics; Ha & Kim 2026,
Applied Sciences; CALIBURN; FADES). What remains distinct, pending a full read of those papers:
(a) capture-level shift inside one environment, with a matched control, instead of cross-dataset shift;
(b) breadth (72 models, 5 seeds) showing seed variance dominates;
(c) the edge setting (16 KB int8 student, cascade) where recalibration must be label-free and cheap;
(d) strict protocol (IAT removed, cross-split de-dup) with measured protocol inflation.
If E1 (recalibration) fails, P1 is a strong evaluation paper with an honest edge-cascade result; that is still
publishable in IEEE IoT-J / TNSM if positioned as an evaluation study.

## 4b. E1 result (run 2026-10-11, approved; `scripts/36_recalibration.py`, `outputs_v3/recalib/recalib.json`)

Decision rule (fixed before the run): median FPR <= 1.5%, median recall loss <= 3 pts, >= 3 families, >= 2 of 3
CICIoMT tasks, at K = 2000 burn-in flows with 1% attacks.

| CICIoMT task | fixed | clean burn-in (conformal) | label-free corrected | naive window | families passing (corrected) |
|---|---|---|---|---|---|
| binary | 2.65% (44/61 > 1.5%) | 0.99% (0/61), loss 0.05 | 1.20% (14/61), loss 0.04 | 0.00%, loss 3.88 | 3 of 5 |
| 6-class | 3.54% (55/72) | 0.99% (0/72), loss 0.11 | 1.38% (31/72), loss 0.07 | 0.00%, loss 3.51 | 3 of 5 |
| 19-class | 2.48% (50/72) | 0.98% (0/72), loss 0.07 | 1.22% (24/72), loss 0.05 | 0.00%, loss 3.82 | 4 of 5 |

- **Rule met** for both the clean burn-in and the label-free method. Robust across K = 500–5000 and attack share
  0–10% (corrected median 1.12–1.52%; p90 tightens with K: ~2.1% at 500, ~1.4% at 5000).
- **Control (CICIoT)**: no harm; corrected 1.17–1.21% vs fixed 1.08–1.10% (small upward bias of the estimator).
- **Where label-free fails**: XGBoost and the ensembles stay at 2.6–5.7%. Their false alarms under shift are
  *confident*: their 0.1%-target threshold inflates to 1.6–5.3% (small MLPs: 0.11–0.47%), so shifted benign flows
  are indistinguishable from attacks by score and the contamination estimate absorbs them.
- **Leaderboard illusion**: at fixed thresholds the ensembles have the top recall (99.83%) but at 6–8% FPR; after
  recalibration every model reaches 99.74–99.78% recall at ~1% FPR. In binary attack detection the large models'
  advantage under capture shift is bought with false alarms.
- **Honest limits**: (1) burn-in flows are a random sample of the new captures (no timestamps), so the clean
  conformal result is valid by construction (exchangeability) — its contribution is the measured sample size
  (K ≈ 1000–2000 benign flows suffice) and the edge cost (one sorted buffer), not the method; (2) temporal drift
  inside a deployment is untested; (3) degenerate-score models (two-stage CatBoost on 6-class, LightGBM on
  19-class/CICIoT) need tie-aware thresholds; (4) KD vs scratch in the far tail: only 19-class significant (p = 0.043).

**Consequence for P1 thesis:** the defensible method claim is "a 16 KB edge student plus a few thousand flows of
benign burn-in (or a label-free contamination-corrected quantile) restores the false-alarm budget after a change of
capture, which the accurate ensembles cannot do without a clean window because their false alarms are
confident." Novelty must still be positioned against Jin & Lee 2026 and Ha & Kim 2026.

## 4c. After the full E-series and the 2026 literature (2026-10-11, evening)

Full texts read: Jin & Lee 2026 (verified: zero recall at source thresholds, 18.9% target FPR), Nguyen et al. 2026
(random stream order, drift signal inactive, no FPR budget), Neto et al. 2023 (CICIoT2023: 10/100-packet windows,
flow-level feature definitions, IAT reported with mean 8.3e7). Added and verified: Biswas 2026 (preprint), Eren et al.
2026 (Electronics), Soylu 2026 (Zenodo artifact), Agbedanu et al. 2026. Matrix: RELATED_WORK_MATRIX.md.

**Revised P1 thesis (narrowed):**
> When an IoT intrusion detector meets traffic from new captures, a false-alarm budget set on validation fails in one
> of two regimes: (i) ranking intact, threshold shifted (CICIoMT2024: ROC-AUC 0.9995, 1% -> 2.5–3.5%), where
> recalibration on deployment traffic restores the budget at ~0.1 pt recall; or (ii) ranking lost or inverted
> (IoT-23: ROC-AUC 0.67 / 0.06, 1% -> 9–39% even when all malware families were seen in training), where no threshold
> strategy, including published feedback-guided adaptation, holds the budget with useful recall. Benign drift alone
> triggers both. A random-sample calibration window overstates what time-ordered deployment achieves. Protocol
> choices (IAT, random split) shift the same pipeline from 68% to 99% accuracy.

Correction (2026-10-12, commissioning screen): the IoT-23 same-family "ranking lost (0.67)" is a pooled-across-captures
artefact; within each capture ROC is 0.96, so that split is a per-capture threshold shift. Genuine ranking loss
remains for the random capture hold-out (within-capture 0.48). A local benign memory fused with the detector gave
1.6–1.9% FPR at 88% recall vs 2.2% at 63% for time-ordered burn-in (2 captures; near miss of the pre-set rule).

What changed from 4b: the E8 method claim is dropped (rule failed on IoT-23); the regime diagnosis (ranking metric +
domain AUC before any recalibration) is the new candidate contribution; benign drift is separated from unseen attacks.

Still open: tree/ensemble seeds on CICIoMT (running), E5 experts on the holdout (resuming), UNSW real-home traffic via
Zeek (after reboot), Mininet (after reboot). CICIoMT has no timestamps, so regime (i) cannot be tested in time order;
a dataset with timestamps and intact ranking would be needed to make a recalibration-method claim.

## 5. Problems a reviewer would find

1. **Test set reuse** (handoff issue #1). The CICIoT/CICIoMT test sets were viewed during specialist, experts,
   gate and rule-search iterations. Fix: fresh holdout (E5) and disclosure.
2. **Pseudo-replication**: "35/40 tasks", "33/40 pairs" treat task × seed as independent. Re-test with 8 tasks as
   units (or hierarchical bootstrap).
3. **Experts seed spread** unmeasured (deterministic XGBoost); 7b running.
4. **Aggregate evasion hides rare-class failure** (50–72% for web attacks).
5. **One shift dataset** (CICIoMT). The shift claim needs a second one (E3).
6. **SHAP fidelity**: arXiv 2512.19488 (SHAP-guided pruning + KD) and AL-Nomasy et al. 2025 (explainable KD IDS)
   occupy SHAP+KD; keep fidelity as a minor result only.
7. **int8 robustness**: Ullah et al. 2026 (CMC) report int8 robustness to PGD on RT-IoT2022. Ours is
   natural corruption and constrained feature-space evasion; different, but not novel enough to headline.
8. **Stale/misleading files**: gate seed-0 summary; LightGBM 0.01% FPR on CICIoT (degenerate scores).
9. **Device claim**: latency is one desktop core, not an IoT device.

## 6. Housekeeping done in this session (outside read-only scope, approved by the user)

- `scripts/35_adversarial.py` + `src/shield/models/xgb.py`: the XGBoost step reloaded a 775 MB model per call and
  ran 5.5 h without output; patched to load once (`load_xgb`). Rerun finished the XGBoost part in 1.7 min.
- Chains `chain_eval.ps1` and `chain_sub.ps1` replaced by `D:\shield_run\outputs_v3\chain_eval2.ps1`
  (adversarial → clipping control → experts subsample). Uncommitted.
