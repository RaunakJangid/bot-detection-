# Experiment decision plan (after the audit)

Nothing below runs without the user's approval. Order = value per unit cost. Each item has a decision rule
fixed **before** the run.

## Running now (approved earlier)

| Job | Purpose | Decision rule |
|---|---|---|
| Clipping control (`30_noise_robustness`, 3 tasks) | Is int8 robustness just input clipping? | If fp32+clipping matches int8 within 0.3 pts on most cells → claim "clipping explains it" and drop int8 robustness as a finding |
| Experts subsample 0.8, 3 seeds (`29_seed_experts`) | Seed spread of +1.62 | Keep the gain only if the 3-seed mean gain > 2 sd and > 0.5 pts |

## Proposed (in order)

### E1 — DONE 2026-10-11: rule met (see FINAL_RESEARCH_AUDIT.md §4b). Next for E1: tie-aware thresholds; time-ordered burn-in on a dataset with timestamps (folds into E3).
### E1 — Label-free recalibration under capture shift (hours, CPU, cached logits)
Data: `outputs_v3/logits/ciciomt_*` validation + test logits, all models.
Methods (all tuned on CICIoMT validation only; test = official test captures, streamed in windows):
1. fixed validation threshold (current), 2. benign-quantile tracking on unlabelled windows (trimmed quantile of
student "benign" scores), 3. split-conformal threshold with a sliding calibration window, 4. EVT tail fit.
Metrics: test FPR vs 1% target, recall, alerts per 1M flows, budget-violation ratio, per model family.
Control: same procedure on CICIoT (no shift) must not hurt.
**Decision:** method kept if median test FPR ≤ 1.5% with recall loss ≤ 3 pts, across ≥ 2 of 3 CICIoMT tasks and
≥ 3 model families. Otherwise P1 is an evaluation paper (C2) without a calibration method.

### E2 — Variance decomposition of FPR transfer (hours)
Mixed model / ANOVA on test FPR with factors family, feature set, distillation, seed. Includes a
size-matched comparison scratch vs KD student with 5+ seeds. Already suggests seed dominates.
**Decision:** if seed explains more variance than family, headline "operating-point reliability is a seed lottery";
recommend ensembling over seeds or E1-style recalibration.

### E3 — Second capture-shift dataset (days)
Preferred: IoT-23 (scenario = capture; real devices). Alternative: CICIoT2023 PCAP re-extraction (risk: one
attack per file confounds capture with class). Same strict protocol, same models (subset).
**Decision:** if inflation does not replicate (median ≤ 1.5%), C2 is limited to IoMT and stated as such.

### E4 — Protocol-inflation ablation (1–2 days)
Factors: IAT kept/removed × cross-split de-dup on/off × random vs strict split; 3 published pipelines
(two-stage CatBoost done; Almahaqeri-style XGBoost; one deep model). Report points lost per factor.
No decision rule needed (descriptive).

### E5 — Fresh holdout (1 day GPU)
Carve ~2M rows from the full CICIoT training pool (never used for selection), retrain the frozen final pipeline
once on the remainder, score once. Report old test as development estimate.

### E6 — Device and testbed (needs hardware)
Raspberry Pi 4/5 + USB power meter for int8 student latency/energy; Mininet+Ryu replay for the cascade.

### E7 — Real benign traffic (optional, 1 week passive)
Home IoT capture → false alarms/day. Blocked until a feature extractor matching CICIoT2023 is confirmed.

## Not to run

Placement variants (P2 only), further metaheuristic tuning, new novelty detectors, stacking, label cleaning,
TabM students, learned gates — all already negative or tied.

## Statistical rules for every new result

Units = dataset-task (not task × seed); ≥ 5 seeds that actually differ; hierarchical bootstrap CIs; report effect
sizes; Holm correction within a family of comparisons; test set touched once per final configuration.
