# SHIELD-IoT

**SHAP-Guided Knowledge Distillation with Hybrid EHO–ACO for Lightweight Botnet Detection and Resilient Security Controller Placement**

The project has two coupled parts:

1. **Detection.** A residual-MLP teacher is trained on CICIoT2023 and Bot-IoT and explained with SHAP. A tiny student (about 3–5k parameters) is then distilled on the SHAP-selected features, using an extra attribution-alignment loss.
2. **Placement.** A hybrid Elephant Herding + Ant Colony optimiser places SDN security controllers so that latency stays low under failures. The student's measured throughput sets each controller's capacity, which links the two parts.

## Setup (GPU desktop)

```powershell
python -m pip install --user uv
uv python install 3.12
uv sync --extra cu128          # CUDA build of PyTorch. On a machine without a GPU: uv sync --extra cpu
```

Copy the two dataset zips to the desktop and set their paths in [configs/paths.yaml](configs/paths.yaml):

| key | file | contents |
|---|---|---|
| `ciciot_zip` | `archive (5).zip` | CICIoT2023, `CICIOT23/{train,validation,test}/*.csv`, about 8M rows, 46 features, 34 classes |
| `botiot_zip` | `archive (6).zip` | Bot-IoT full, `data_1..74.csv`, about 73M rows |

## Running

Every script is resumable: finished runs are skipped, and `--force` recomputes them. `--smoke` runs on tiny data and writes to separate roots (`data_smoke/`, `outputs_smoke/`).

```powershell
uv run python scripts/run_all.py --smoke     # about 5-15 min end to end; checks every expected output exists
uv run python scripts/run_all.py             # the full pipeline
```

To run the stages one at a time (times are estimates for an i7-12700F + RTX 3060):

| step | command | est. time |
|---|---|---|
| ingest zips to Parquet | `uv run python scripts/01_ingest.py` | ~30-60 min |
| clean, de-duplicate, split, scale | `uv run python scripts/02_preprocess.py` | ~20-30 min |
| teachers (5 tasks x 5 seeds) | `uv run python scripts/03_train_teacher.py` | ~10-15 h |
| XGBoost reference | `uv run python scripts/03b_xgboost.py` | ~2-4 h |
| SHAP + stability | `uv run python scripts/04_shap.py` | ~1-2 h |
| teacher cache for KD | `uv run python scripts/05_cache_teacher.py` | ~15 min |
| distillation ablations | `uv run python scripts/05_distill.py` | ~8-15 h |
| F1-vs-k sweep | `uv run python scripts/05_distill.py --k-sweep` | ~2-4 h |
| latency / size / FLOPs | `uv run python scripts/06_latency.py` | ~15 min |
| placement benchmark | `uv run python scripts/07_placement.py` | ~6-20 h (CPU, 16 processes) |
| coupling experiment | `uv run python scripts/07b_coupled.py` | ~1-2 h |
| tables + figures | `uv run python scripts/08_make_figures.py` | ~2 min |

`07_placement.py` only uses the CPU. You can start it while steps 3–5 use the GPU; it then uses the default detector throughput (`default_mu_flow`). To use the measured student throughput instead, run it after `06_latency.py`.

**Quick pass first.** Run `03`–`05` with `--seeds 0` and `07` with a reduced grid (edit `configs/placement.yaml`) to check that the method works before committing to the full grid.

Useful flags: `--dataset ciciot|botiot`, `--task binary`, `--seeds 0 1`, `--variants shield kd_all`, `--jobs 3`, `--cpu`, `--workers 16`, and `07_placement.py --analyse-only`.

## Outputs

```
outputs/
  data_report_<ds>.json                     class counts per split, de-dup statistics
  teacher/<ds>_<task>_resmlp_s<seed>/        model.pt, metrics.json, confusion.npy
      shap/                                  importance.json (ranking, k95), shap_values.npy, plots
      cache/                                 fp16 teacher logits + grad x input (KD teacher only)
  xgboost/<ds>_<task>_s<seed>/
  kd/<ds>_<task>/<variant>_s<seed>/          student.pt, student_int8.pt, metrics.json (incl. int8, fidelity)
  latency/latency.json                       single-core CPU latency/throughput, size, FLOPs, GPU throughput
  placement/                                 runs.jsonl/parquet, optimum.jsonl, summary.csv, wilcoxon.csv, friedman.json
  coupled/                                   coupled_all.csv, min_controllers.csv
  paper/tables/*.csv|tex, paper/figures/*.pdf|png
```

## Method notes

**Data**
- CICIoT2023 keeps its provided split and has three label granularities: binary, 8 families and 34 classes.
- Bot-IoT drops identity and time columns (`pkSeqID`, `stime`, `ltime`, addresses, MACs, OUIs, ports) and one-hot encodes `proto`, `flgs` and `state`.
- Exact duplicate feature rows are removed globally by hash. Hashes with conflicting labels are dropped entirely.
- Bot-IoT uses a 70/15/15 split stratified by subcategory, with binary and 5-category tasks.
- Both datasets use median imputation, a signed-log transform and standardisation, all fitted on train only.

**Teacher.** A residual MLP (about 1.2M parameters, 4×384) trained with AdamW, warmup + cosine schedule and bf16 autocast, early-stopped on validation macro-F1. Bot-IoT uses √-balanced sampling with 8M samples per epoch.

**SHAP.** `GradientExplainer` with 1,000 stratified background rows and 5,000 explained test rows. Global importance is the mean |SHAP| for each row's true class.

**SHIELD loss.** `α·CE + β·T²·KL(teacher‖student) + γ·(1 − cos(w⊙A_s, w⊙A_t[S]))`, where `A` is the true-class gradient × input. `S` is the SHAP top-k feature set, and `w` is the normalised global SHAP importance on `S`.

**Ablations.** `scratch_all`, `kd_all`, `kd_shap`, `shield`, `kd_random`, `kd_mi` and `kd_variance`, plus int8 dynamic quantisation of every student.

**Placement objective.** Capacitated greedy-regret assignment with a backup controller for every switch. The objective is a weighted sum of:
- average latency (propagation plus M/M/1 sojourn),
- maximum latency,
- inter-controller latency,
- load imbalance,
- the worst mean latency over single-controller failures,

each normalised by its mean over random placements, plus a penalty for overload in normal and failure states. The core runs in numba.

**Hybrid EHO–ACO.** EHO clans do the exploitation. ACO ants (MAX–MIN pheromone, demand-closeness heuristic, coverage mask) replace EHO's random separating operator. Matriarchs and the global best deposit pheromone, and the global best periodically gets a 1-swap local search.

**Baselines.** All use the same evaluation budget: GA, set-based PSO, SA, pure EHO, pure ACO, random search, greedy k-median, greedy k-center, weighted k-means, PageRank, and a capacitated k-median ILP solved with HiGHS. For small instances, brute force gives the true optimum.

**Resilience.** Evaluated after optimisation:
- all single-controller and sampled double-controller failures;
- random and betweenness-targeted link and node failures at 5–20%.

**Coupling.** Demand is fixed relative to the teacher's capacity. For each detector (teacher, SHIELD student, int8 student), the experiment finds the smallest k that keeps at least 95% of switches within the SLA, with no overload, also under the worst single failure.

## Tests

```powershell
uv run pytest
```
