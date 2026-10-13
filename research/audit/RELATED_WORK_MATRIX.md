# Related work matrix

Verification key: **V-CR** = metadata confirmed via Crossref API this session; **V-WEB** = confirmed via
publisher/arXiv/PubMed page this session; **KNOWN** = standard reference, DOI not re-checked this session;
**UNVERIFIED** = could not be found; do not cite.

## Closest to the revised P1 thesis (operating point under shift)

| Work | Venue | Verif. | What it does | Overlap with us | What stays ours |
|---|---|---|---|---|---|
| Jin & Lee, "Benchmark-Shift-Aware Intrusion Detection for Evolving Network Traffic: Cross-Dataset Generalization, Calibrated Alerting, and Score-Orientation Diagnostics" | Electronics 15(17):3761, 2026, doi 10.3390/electronics15173761 (peer-reviewed) | **V-FULLTEXT** (PDF supplied by user, read 2026-10-11) | HIKARI-2021, CICIDS2017, CICIoT2023 subset (Merged01-03, 97.7% attacks) in harmonised Rich-64 / Minimal-13 spaces; XGBoost, Transformer, SSL Transformer; 5% FPR thresholds from source validation; label-free target-score quantile budgets; score-orientation (oracle-flip) diagnostic; CICIDS2017 day-split. Verified numbers: zero recall at source thresholds HIKARI->CICIDS2017; XGBoost 18.9% target FPR in reverse; Friday FPR 7.27% (XGB) | **High**: threshold transfer, alert budgets, orientation inversion, chronological check | Their limits (their Sec. 4.7): single seed; CICIoT only as a subset; chronology only on CICIDS2017; no efficiency; no recalibration (listed as future work: analyst-validated labels + drift-triggered threshold re-estimation). Ours: capture-level shift inside one dataset with random-split controls; 1% and 0.1% budgets; 61–72 models x seeds; benign-drift vs unseen-attack separation (IoT-23 same-family split); time-ordered vs random recalibration; controlled comparison incl. their proposed future method |
| Ha & Kim, "Operationally Constrained Zero-Day Intrusion Detection with Target-FPR Calibration and Similarity Graph Construction" | Applied Sciences 16(5):2284, 2026, doi 10.3390/app16052284 (peer-reviewed) | V-CR | Threshold at target FPR on benign validation, transferred to OOD (zero-day) test; GNNs show heavy-tailed false-alarm amplification | **High** for the protocol (target-FPR transfer) | Shift type (benign capture drift, not attack novelty); model breadth; edge |
| Nguyen, Ta, Dao, Nguyen, Phung, "Feedback guided decision boundary adaptation for stable IoT intrusion detection under distribution shift" | Discover Artificial Intelligence 6:1394, 2026, doi 10.1007/s44163-026-02370-1 (peer-reviewed, OA) | **V-FULLTEXT** | Fixed RF + EMA abnormality gate on d = 4 features; thresholds updated every 200 samples from mean entropy and a Hoeffding drift signal; RT-IoT2022, IoTID20, CICIoT2023 (57.5k-row samples). F1 0.9387 vs ARF 0.9407, FPR 3.12% vs 3.49% | **High** for E8 (threshold adaptation) | Their streams are randomly re-ordered per seed ("not verified wall-clock chronology"); shift is injected at batch 120 and their drift signal stays 0 throughout (Fig. 4); no FPR budget; gate features unnamed. We re-implemented it (42_e8_stream.py) and compare on real time-ordered captures |
| Agbedanu et al., "Confidence-driven lightweight adaptive online deep learning for IoT intrusion detection" | Discover Telecommunications 2026, doi 10.1007/s44595-026-00001-9 | V-WEB (abstract) | Online learning + adaptive feature gate + quantisation; Raspberry Pi 5; drift via synthetic injection | Medium (edge + adaptation) | Real capture shift; operating-point focus; but they have real-device numbers and we do not |
| Biswas, "The Evaluation Protocol Is the Hidden Variable: A Drift-Aware Re-Examination of Flow-Based Intrusion Detection Benchmarks" | Preprints.org 2026, doi 10.20944/preprints202606.0903.v1 (**preprint, not peer-reviewed**) | V-WEB (search summary; page 403) | CSE-CIC-IDS2018 and CIC-DDoS2019; protocol is the dominant variable; 36/45 features severe drift (PSI >= 0.25); time-ordered calibration under transfer. The 0.79–0.82 vs 0.02 macro-F1 figures quoted by a review could NOT be verified | **High** for protocol + temporal evaluation | Not IoT; we add low-FPR operating points, benign drift vs unseen attacks, recalibration comparison |
| Eren, Küçük, Saleh, Konyar, Hardy, Khan, "Distributional Drift in IoT Intrusion Detection Systems: Implications for Cross-Dataset Generalisation" | Electronics 15(11):2307, 2026, doi 10.3390/electronics15112307 (peer-reviewed) | V-CR | BoT-IoT, ToN-IoT, UNSW-NB15; distributional inconsistencies and feature biases degrade cross-dataset performance; harmonisation and feature selection | Medium (cross-dataset drift) | We study operating points under capture shift within datasets, not feature-space harmonisation |
| Soylu, "Benchmark-to-Deployment Gaps in Intrusion Detection System Evaluation Under Temporal and Telemetry Shift" | Zenodo 2026, doi 10.5281/zenodo.21419353 (**deposited artifact, resource type "model"; peer review unverified**) | V-WEB | UNSW-NB15, CICIDS2017, ThreeFold testbed; evaluation ladder; "cross-capture operating thresholds are not portable": RF keeps PR-AUC 1.000 while source thresholds give zero/low recall | **High** (threshold portability across captures) | Cite as technical artifact only. Not IoT; no benign-drift separation; no recalibration comparison |
| CALIBURN (Youssef; arXiv 2605.24696, v3 retitled "Stream Assembly Is an Uncontrolled Treatment in Streaming IDS Benchmarks") | arXiv 2026 | V-WEB | Conformal risk control in streaming IDS (v1); v3 is a benchmark-methodology critique on CICIDS2017 | Medium | Not IoT, no edge |
| FADES: Adaptive Drift Estimation via Conformal Signals for Streaming IDS | Electronics, doi 10.3390/electronics15102114 | V-WEB (search result) | Conformal drift signals, streaming | Medium | Not checked in depth |
| Li, "Robust Conformal Intrusion Detection via Traffic-Aware Calibration and Attack-Orbit Invariance" | arXiv 2609.19241, 2026 | V-WEB | Conformal coverage under adversarial perturbation; LM-based NIDS | Low–medium (adversarial, not capture shift) | Edge, natural shift |
| Transcend (Jordaney et al.) | USENIX Security 2017 | KNOWN | Conformal evaluation to detect concept drift in malware classifiers | Concept: drift-aware rejection | Domain (IoT NIDS), FPR budget |
| TESSERACT (Pendlebury et al.) | USENIX Security 2019 | KNOWN | Temporal/spatial bias in malware evaluation | Concept: evaluation bias | IoT capture shift |
| Arp et al., "Dos and Don'ts of ML in Computer Security" | USENIX Security 2022 | KNOWN | Pitfalls incl. data snooping, base-rate fallacy | Supports our protocol arguments | — |

## Datasets, protocol and cross-domain

| Work | Venue | Verif. | Relevance |
|---|---|---|---|
| Neto et al., CICIoT2023 | Sensors 23(13):5941, 2023 | KNOWN | Dataset; RF 34-class 71.4 with IAT |
| Dadkhah et al., CICIoMT2024 | Internet of Things, 2024 (PII S2542660524002920) | V-WEB (link in report) | Dataset; capture-level test files |
| Doménech, León, Siddiqui, Pegueroles, "Evaluating and enhancing IDS in IoMT: the importance of domain-specific datasets" | Internet of Things 32:101631, 2025 | V-WEB | CICIoT2023 ↔ CICIoMT2024 cross-dataset F1 drops up to 66.9%; windowing, splits. **Overlaps our item 32** (cross-dataset) — cite, do not claim cross-dataset collapse as new |
| Almahaqeri et al., "An optimized gradient boosting framework for IoT intrusion detection: CICIoT2023" | Scientific Reports, 2026, doi 10.1038/s41598-026-47399-5 | V-WEB | 34-class 88.8 with IAT, no de-dup — protocol-inflation example |
| Rodríguez, Mehavilla, Alesanco, García, "A rule-based label harmonization framework for cross-dataset IoT intrusion detection" | Internet of Things 39, 2026 (PII S2542660526001605) | V-WEB | Cross-dataset F1 < 0.50 except MLP — consistent with ours |
| Chhetry, Dutta, Matam, Barbhuiya, "CROSS-IoT: A Zeek-Driven Data Harmonization Pipeline" | Int. J. Communication Systems 39(12), 2026, doi 10.1002/dac.70565 | V-CR | Five datasets unified by Zeek; 95–99% cross-dataset accuracy. Counterpoint: harmonised features transfer |
| IoT-23 (Garcia, Parmisano, Erquiaga) | Zenodo 2020 | KNOWN | Candidate second capture-shift dataset (E3) |
| Engelen et al., CICIDS2017 troubleshooting | IEEE SPW 2021 | KNOWN | Precedent for dataset-flaw papers |

## Edge / distillation / quantisation (P1 supporting results; not novelty)

| Work | Venue | Verif. | Overlap |
|---|---|---|---|
| Ullah, Wu, Kamal, Alzaylaee, Mohamed, "Quantized Intrusion Detection for Resource-Constrained IoT: Efficiency and Adversarial Robustness" | CMC 88(3), 2026, doi 10.32604/cmc.2026.084409 | V-CR + V-WEB | 1D-CNN FP32/FP16/INT8 TFLite on RT-IoT2022; INT8 97.5% under PGD; Cortex-M4 estimates. **Overlaps "int8 robustness"** — ours differs (natural corruption, constrained evasion, integer-only MLP) |
| Ullah, Wu, Kamal, Saudagar, "KronNet" | Scientific Reports 15:20850, 2025, doi 10.1038/s41598-025-08921-3 | V-CR | Lightweight Kronecker FFN IDS |
| Benaddi et al., "Lightweight IDS in IoT via SHAP-Guided Feature Pruning and Knowledge-Distilled Kronecker Networks" | arXiv 2512.19488, 2025 | V-WEB | **Directly occupies SHAP-guided KD** (TON_IoT, macro-F1 > 0.986) |
| AL-Nomasy, Alamri, Aljuhani, Kumar, "Transformer-based knowledge distillation for explainable IDS" | Computers & Security 154:104417, 2025 | V-CR | Explainable KD IDS (the "DistillGuard" label in the handoff is not its title) |
| Umair et al., IEEE TCE 2025 (KD + SHAP) | — | **UNVERIFIED** | Not found in two searches; do not cite until a DOI is supplied |
| Jacob et al., integer-only quantisation | CVPR 2018 | KNOWN | Method basis for IntMLP |
| Gorishniy et al., TabM | ICLR 2025 | KNOWN | Deep tabular baseline |

## Methods we use (cite as tools)

Hinton et al. 2015 (KD); Menon et al. ICLR 2021 (logit adjustment); Caruana et al. ICML 2004 (ensemble selection);
Cover & Hart 1967 (NN error bound); Lee et al. NeurIPS 2018 (Mahalanobis OOD); Dasgupta et al. Science 2017 (FlyHash);
Lundberg & Lee NeurIPS 2017 (SHAP); Mirsky et al. NDSS 2018 (Kitsune, edge anomaly detection baseline).

## Targeted gap search (2026-10-12)

| Work | Verif. | What it shows | Effect on our claims |
|---|---|---|---|
| Ahmed, Rahaman, Roy, Alam, "The dashboard says 'perfect': a systematic audit of label-free model monitoring and test-time adaptation under temporal distribution shift in network intrusion detection", Frontiers in AI, 8 Oct 2026, doi 10.3389/frai.2026.1928858 | V-WEB (full text) | CIC-IDS2017 <-> CSE-CIC-IDS2018 (1-year shift): 16 adaptation variants (incl. quantile alignment) never beat the unadapted model in both directions; ATC predicts accuracy 1.000 for a model with zero attack recall; conformal abstention is the only label-free signal correct in all 12 trials | **Closest overlap for Case 2 and for any label-free "which case" check.** Not IoT; no benign-drift separation; no FPR budget; no case where recalibration works. Our regime check MUST be benchmarked against ATC and conformal abstention |
| Bouke, Sayeed, Heng, Abdullah, Othman, "Multi-Level Distributional Entropy from Flow Summary Statistics for Explainable NIDS", arXiv 2606.29797 (rev. Oct 2026) | V-WEB (abstract) | Names "threshold-ranking divergence": AUC 0.84–0.87 kept while fixed-threshold detection collapses to 0.08 in a 703k-flow pseudo-live replay (NSL-KDD, CICIDS2017/18, UNSW-NB15) | **Our Case 1 has a name in the literature.** They do not test recalibration; we show when it works (CICIoMT, IoT-23 same-family, real home) and when it cannot (ranking lost) |
| "Calibration Granularity, Not Contamination: Diagnosing a TCN Anomaly Detector's False Positive Advantage in Cross-Dataset IoT Traffic", Future Internet 18(9):447, 2026, doi 10.3390/fi18090447 | V-WEB (search summary; page 403) | TCN-VAE trained on benign BoT-IoT: cross-dataset FPR 22% / 48% / 65% (BoT-IoT, IoT-23, ToN-IoT); calibration granularity explains FP differences | Related to our per-capture threshold finding; read full text before claiming per-capture calibration |
| "Detecting Unseen IoT Attacks with Calibrated Dual Evidence Under Low False-Positive Budget", Entropy 28(9):1026, 2026 | V-WEB (search summary; page 403) | IoT-23 family hold-out at ~1% FPR (0.96%), unseen recall 85.8 -> 90.9% | Overlaps low-FPR IoT-23 evaluation; theirs is family hold-out within captures, ours capture hold-out + benign drift. Read full text |
| Confidence-gated cascade IDS (Sci Rep 2025, doi 10.1038/s41598-025-34124-x); offloading ~10% to cloud (Sensors 26:356) | V-WEB (search) | Cascades / offloading with fixed escalation shares | Nobody we found measures escalation load blowing up under shift -> our E6 point remains open |
| SDN/Mininet detect-and-mitigate (arXiv 2104.07332; replay-attack SDN-IoT 2026; NGNDAI 2025) | V-WEB (search) | Drop-rule mitigation in Mininet is common; detection/mitigation latency rarely quantified with a validated feature space | Detect-and-block is supportive, not novel |

## P2 gap check: detection + controller placement (full texts read 2026-10-13)

| Work | Verif. | What it does | What it leaves open (our P2 gap) |
|---|---|---|---|
| Seyedkolaei & García, "Optimized Placement of Security Controllers for Enhancing Intrusion Detection and Resilience in Software-Defined IoT Networks", J. Netw. Syst. Manag. 34:110, May 2026, doi 10.1007/s10922-026-10062-5 | V-FT | NSGA-II, weighted security (attacker proximity, attack-type severity) - delay - cost; primary + backup controller per IoT node; fixed controller capacity (alpha = 4000 KB, beta 250 KB or 100-400 KB "dynamic"); BtAsiaPac/Chinanet/Colt/Cogentco subsets; baselines random, greedy, ILS; MATLAB | No detector model at all: capacity is a constant, never derived from a detector's cost; "dynamics" = node count / packet-size changes with re-assignment, not detection-load shift; no exact optimum, no strong metaheuristic baselines. Note: reported SDs equal 5.0% of the mean for delay, cost and load in every cell |
| Sapkota, Ray, Yadav, Dawadi, Joshi, "Machine Learning-Based Attack Detection and Mitigation with Multi-Controller Placement Optimization over SDN Environment", J. Cybersecur. Priv. 5:10, Mar 2025, doi 10.3390/jcp5010010 | V-FT | K-means++ / OPTICS placement by WCSS of distance/bandwidth (Aarnet, Savvis); central XGBoost IDS on a self-generated Mininet flow-stats dataset (436k rows, time-sorted then 80/20 split, undersampling), 98.5% accuracy; detect-and-block in 0-10 s (bounded by 10 s stats polling) | IDS deliberately decoupled from controllers, so detector cost never enters placement; no capacity or latency SLA; simple floods only; no shift. Supports our framing that detection load and placement are treated separately |
| Sebopelo & Isong, "An Integrated Framework for Controllers Placement and Security in Software-Defined Networks Ecosystem", J. Inf. Syst. Informatics 6(1), Mar 2024, doi 10.51519/journalisi.v6i1.682 | V-FT (pp. 1-3) | Enhanced-DBSCAN placement side by side with a kNN IDS (100% on a "LAN dataset") | Placement and IDS built side by side, not coupled |
| Killi & Rao, "Optimal Model for Failure Foresight Capacitated Controller Placement in Software-Defined Networks", IEEE Commun. Lett. 20(6):1108, 2016, doi 10.1109/LCOMM.2016.2550026 | V-FT | FFCCP ILP: K closest reference controllers, shared capacity, min worst-case latency to the K-th reference; AT&T/GEANT/SURFNET, loads U(100k,400k), U = 7800 kpps, CPLEX | The published benchmark model; reproduced faithfully in scripts/57_killi_rao.py (row 28af) |
| Mohanty & Sahoo, "Metaheuristic algorithms for capacitated controller placement in SDN considering failure resilience", CCPE 36(24):e8254, 2024 | V-WEB (abstract) | Enhanced GA, PSO, hybrid PSO-SA, GWO on the failure-foresight capacitated model, four Topology Zoo networks | Full text still needed to compare numbers |

**P2 position (2026-10-13):** no paper found derives controller capacity from the deployed detector's measured cost or models the escalation load changing under traffic shift. Metaheuristics for CPP and "IDS + placement" frameworks are crowded and are not P2's novelty.

## Novelty position after the 2026 literature (updated 2026-10-11)

Not claimable as new: that thresholds fail under shift (Jin & Lee; Ha & Kim; Soylu), that the evaluation protocol
dominates reported scores (Biswas; our E4 only quantifies it on IoT datasets), that adaptive thresholds help
(Nguyen et al.), that compact models run at the edge (Agbedanu et al.; Ullah et al.), cross-dataset drift (Eren et al.).
Candidate specific contributions, each to be supported by our data only:
1. Two failure regimes under real IoT capture change, diagnosed with ranking metrics: CICIoMT keeps ranking
   (test ROC-AUC 0.9995) so the failure is the threshold; IoT-23 loses ranking (0.67 same-family) or inverts it
   (0.06), so no threshold method can help. Recalibration is only meaningful in the first regime.
2. Benign drift alone (same malware families, new captures) breaks a 1% budget (9–39%), separated from unseen attacks.
3. Random-sample recalibration overstates time-ordered deployment (IoT-23 random-capture hold-out: burn-in 0.6–0.75% random vs 3.2–4.4% time-ordered on the same captures), and when ranking is lost no threshold method (incl. a re-implementation of Nguyen et al.) holds a 1% budget with useful recall (E8).
4. Seed variation is comparable to architecture differences for FPR transfer and family-dependent (trees stable and consistently high; small MLPs low on median but seed-unstable) — checked with tree/ensemble seeds (43_tree_seeds.py).
5. CICIoT2023 window-size artefact: Number/Weight take three values tied to the 10/100-packet windowing.

## Reading still required before writing

1. Jin & Lee 2026 full text (closest; decides how C2/C3 are positioned).
2. Ha & Kim 2026 full text (+ their GitHub reimplementation).
3. FADES and CALIBURN v1 (conformal methods we would compare against in E1).
4. Doménech et al. 2025 full text (CICIoMT protocol details).
