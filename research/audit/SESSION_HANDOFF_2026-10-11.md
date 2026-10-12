# Session handoff, 2026-10-11

Full chat transcripts (raw JSONL) and assistant memory are backed up in `D:\shield_run\session_history\`
(local only; they contain local paths and account details, so they are not committed).
The experiment ledger is `results/v3/EXPERIMENT_LOG.md` (43 experiments, PASS/FAIL/PEND).

## Rules agreed with the user
- Strict protocol only: no leakage, no change of split protocol for better numbers, choices on validation, test once.
- Report first; run anything only after the user approves it. Do not write the paper yet.
- Usage limit is exhausted: queue jobs, report only when asked for "status".
- `configs/paths.yaml` and `configs/kd.yaml` stay local and uncommitted. Mininet/Ryu after the Monday reboot.

## Jobs queued/running at handoff (do not touch)
| Chain | What | Log |
|---|---|---|
| chain_eval.ps1 | 35_adversarial (XGBoost part pending), then 30_noise_robustness clipping control (3 tasks) | adversarial.log, noise_clip.log |
| chain_sub.ps1 | 29_seed_experts --subsample 0.8, 3 seeds (seed spread for the +1.62 experts gain) | seed_experts_sub.log |
(logs under the v3 outputs directory)

## Known issues to carry into the full audit
1. The test set was viewed during many iterations (specialist group, experts, gate); final choices were made on
   validation, but the test set is not strictly untouched. Needs disclosure or a fresh holdout.
2. p-values for the decision rule (35/40 tasks) and SHAP fidelity (33/40 pairs) treat non-independent tasks/seeds
   as independent. Need seed- or dataset-level analysis.
3. Experts +1.62: first 3-seed run was deterministic (identical seeds); spread pending (chain_sub).
4. IAT exclusion: evidence supports "collection-specific shortcut" (CICIoMT two-stage 78 -> 92 with IAT), not proven
   direct label leakage. Word it that way.
5. "v2: 14.9" in the log = old dynamic-quantisation loss, not a current result.
6. int8 robustness claim waits for the clipping control; check overlap with CMC 2026 (doi 10.32604/cmc.2026.084409).
7. Capture-grouped split is impossible on CICIoT2023 CSVs (no capture IDs); CICIoMT already has capture-level test.
8. GWO+SA is published (hGWO-SA); EHO-ACO only ties. Placement belongs in a separate paper (P2) or limitations.

## Plan for the full audit (approved for "tomorrow")
Read-only. Create in `research/audit/`: FINAL_RESEARCH_AUDIT.md, RELATED_WORK_MATRIX.md, EXPERIMENT_DECISION_PLAN.md,
REPRODUCIBILITY_AUDIT.md, CLAIMS_AND_LIMITATIONS.md, BIBLIOGRAPHY.bib. Verify every DOI before citing.
Papers to read first: Almahaqeri 2026 (Sci Rep), Rodriguez 2026 (IoT), CMC 2026 quantized IDS robustness,
CROSS-IoT (doi 10.1002/dac.70565), IoMT domain-shift (doi 10.1016/j.iot.2025.101631), Umair et al. TCE 2025,
DistillGuard (doi 10.1016/j.cose.2025.104417), KronNet (doi 10.1038/s41598-025-08921-3), arXiv 2512.19488.
Mixed-dataset (CICIoT + CICIoMT shared classes) experiment: write the protocol only, do not run.
