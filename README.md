# SHIELD-IoT

**SHAP-Guided Knowledge Distillation with Hybrid EHO–ACO for Lightweight Botnet Detection and Resilient Security Controller Placement**

The project has two coupled parts:

1. **Detection.** A residual-MLP teacher is explained with SHAP. A tiny student (about 3–5k parameters) is then distilled on the SHAP-selected features, using an attribution-alignment loss. It's evaluated on three datasets under a leakage-free protocol, and across networks (CICIoT2023 ⇄ CICIoMT2024).
2. **Placement.** A hybrid Elephant Herding + Ant Colony optimiser places SDN security controllers so that latency stays low under failures. The student's measured throughput sets each controller's capacity, which links the two parts.

## Setup (GPU desktop)

```powershell
python -m pip install --user uv
uv python install 3.12
uv sync --extra cu128          # CUDA build of PyTorch. On a machine without a GPU: uv sync --extra cpu
```

Set the dataset paths in [configs/paths.yaml](configs/paths.yaml). Leave all three zips zipped; the code reads them directly.

| key | file | contents |
|---|---|---|
| `ciciot_zip` | `archive (5).zip` | CICIoT2023, `CICIOT23/{train,validation,test}/*.csv`, ~8M rows, 46 features, 34 classes |
| `ciciomt_zip` | `ciciomt2024.zip` (GitHub release `datasets-v1`) | CICIoMT2024 Wi-Fi/MQTT attacks, `CICIoMT2024/{train,test}/*.pcap.csv`, 45 features, 18 attacks + benign; the train/test split is by capture |
| `botiot_zip` | `archive (6).zip` | Bot-IoT full, `data_1..74.csv`, ~73M rows |

## Datasets and protocol ([configs/datasets.yaml](configs/datasets.yaml))

| dataset | group | built from | what it is |
|---|---|---|---|
| `ciciot`, `ciciomt`, `botiot` | main | each zip | **Strict protocol** (main results). Leaky features removed: `IAT` from both CIC datasets, since it encodes capture time, and `seq` from Bot-IoT, since it encodes record order. Exact duplicate feature vectors are removed across splits, so no test row also appears in train. |
| `xciciot`, `xciciomt` | cross | both CIC zips | The 43 features and 5 classes (Benign, DDoS, DoS, Recon, Spoofing) both CIC datasets share, used for cross-dataset transfer |
| `ciciot_std`, `ciciomt_std`, `botiot_std` | standard | each zip | Original feature set, seed 0 only. Used only to measure how much the leaky features inflate scores, and for the leakage audit. |

Tasks:
- CICIoT2023: binary / 8 families / 34 classes
- CICIoMT2024: binary / 6 categories / 19 attacks
- Bot-IoT: binary / 5 categories
- Shared datasets: binary / shared5

## Running

Every script is resumable: finished runs are skipped, and `--force` recomputes them. `--smoke` runs on tiny data and writes to separate roots (`data_smoke/`, `outputs_smoke/`). `--dataset` accepts `all`, `main`, `cross`, `standard`, a dataset name, or a comma-separated list.

```powershell
uv run python scripts/run_all.py --smoke     # end-to-end on tiny data; checks every expected output exists
uv run python scripts/run_all.py             # the full pipeline
```

| step | command | what it does |
|---|---|---|
| ingest | `scripts/01_ingest.py` | zips → Parquet |
| preprocess | `scripts/02_preprocess.py` | strict/standard/shared datasets, cross-split de-dup, scaling fitted on train |
| leakage audit | `scripts/02b_leakage_audit.py` | macro-F1 of every feature alone, on the standard datasets |
| teachers | `scripts/03_train_teacher.py` | residual MLP, 5 seeds (main), 3 (cross), 1 (standard) |
| XGBoost | `scripts/03b_xgboost.py` | strong tabular reference |
| tiny baselines | `scripts/03c_baselines.py` | depth-10 decision tree + logistic regression |
| SHAP | `scripts/04_shap.py` | true-class GradientExplainer, ranking stability |
| teacher cache | `scripts/05_cache_teacher.py` | fp16 logits + gradient × input for KD |
| distillation | `scripts/05_distill.py` | 7 ablation variants + int8; `--k-sweep` for F1 vs k |
| latency | `scripts/06_latency.py` | single-core CPU latency/throughput, size, FLOPs |
| placement | `scripts/07_placement.py` | 12 algorithms × topologies × (k, ρ) × 30 seeds + resilience |
| coupling | `scripts/07b_coupled.py` | controllers needed per detector |
| cross-dataset | `scripts/09_cross_dataset.py` | zero-shot (source vs target normalisation), few-shot curve, never-seen attacks, SHAP agreement |
| paper | `scripts/08_make_figures.py` | all tables (CSV + LaTeX) and figures (PDF + PNG) |

**Run a quick pass first.** Run detection with `--seeds 0` and placement with a reduced grid, to check the method before the full grid.

## Outputs

```
outputs/
  data_report_<dataset>.json                 class counts, de-dup statistics, dropped features
  leakage/<dataset>_<task>.csv               single-feature audit
  teacher/<ds>_<task>_resmlp_s<seed>/        model.pt, metrics.json, confusion.npy, shap/, cache/
  xgboost/, baselines/                       reference models
  kd/<ds>_<task>/<variant>_s<seed>/          student.pt, student_int8.pt, metrics.json (int8, SHAP fidelity)
  latency/latency.json
  placement/                                 runs, optimum, summary, wilcoxon, friedman (+ Nemenyi CD)
  coupled/                                   controllers needed per detector
  cross/                                     per-block JSON + zero-shot / few-shot / unseen summaries
  paper/tables/*.csv|tex, paper/figures/*.pdf|png
```

## Method notes

**Leakage controls.** These follow [Arp et al., USENIX Security '22](https://www.usenix.org/system/files/sec22summer_arp.pdf) and recent IoT-IDS shortcut audits:
- identity and time columns are removed;
- known artifact features are removed (strict protocol);
- duplicates are removed across splits;
- scalers are fitted on train only;
- a single-feature leakage audit is reported;
- the inflation from the leaky features is reported next to the honest numbers.

**Teacher and SHAP.** The teacher is a residual MLP (4×384). SHAP uses `GradientExplainer` and explains only each row's true-class output. Global importance is the mean |SHAP|.

**SHIELD loss.** `α·CE + β·T²·KL(teacher‖student) + γ·(1 − cos(w⊙A_s, w⊙A_t[S]))`, where `A` is the true-class gradient × input, `S` is the SHAP top-k feature set, and `w` is the normalised SHAP importance.

**Detection ablations.** `scratch_all`, `kd_all`, `kd_shap`, `shield`, `kd_random`, `kd_mi`, `kd_variance`, plus int8. Variants are compared with a Friedman test and a Nemenyi critical-difference diagram over the dataset/task/seed blocks.

**Cross-dataset (CICIoT2023 ⇄ CICIoMT2024).** Both use the same CIC feature extractor; Bot-IoT is excluded because its flow exporter differs. Each direction measures:
- zero-shot transfer, with source-statistics vs unlabelled target-statistics normalisation;
- few-shot fine-tuning at 0.1/1/5/10% labelled target data, against the same student trained from scratch;
- whether never-seen attack families (MQTT; Mirai/Web/BruteForce) are still flagged as attacks.

**Placement.** Capacitated greedy-regret assignment with backup controllers. The objective is a weighted sum of average and maximum latency (propagation + M/M/1), inter-controller latency, load imbalance and worst single-failure latency. Hybrid EHO–ACO is compared with GA, PSO, SA, pure EHO and pure ACO, random search, k-median, k-center, k-means, PageRank, a capacitated k-median ILP (HiGHS), and the brute-force optimum. Statistics: 30 runs, Wilcoxon + Holm, Friedman + Nemenyi critical difference. Resilience covers controller, link and node failures.

## Tests

```powershell
uv run pytest
```
