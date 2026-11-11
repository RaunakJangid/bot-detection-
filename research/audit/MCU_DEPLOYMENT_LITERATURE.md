# On-device (MCU) deployment of IoT intrusion detectors: literature analysis (2026-10-21)

Purpose: plan the nRF52832 test of the 16 KB int8 SHIELD student and position it against prior work.
"Full text" = read completely; "abstract" = only abstract/snippet available (publisher blocked).

## A. Intrusion-detection papers that claim MCU / TinyML deployment

| Paper | Read | Hardware actually used | Measured or estimated | What they measured | Data / protocol | Relevance to us |
|---|---|---|---|---|---|---|
| Ullah et al., CMC 88(3) 2026 (doi 10.32604/cmc.2026.084409) | full text | none; "Cortex-M4 @160 MHz (STM32F429)" analytical model | **Estimated** (T = N_ops/(f*IPC); energy model with unstated constants); board testing listed as future work | INT8 1D-CNN 13.6k params, 25.7 KB; est. 0.083 ms, 0.0083 mJ | CICIoT2023, 200k random rows, random 80/20 split, IAT not discussed; 99.1% acc | Same dataset + similar size as ours; shows the field's typical gap: no device, leaky split. Our test gives measured numbers for a comparable model |
| Kandil et al., IJCA 187(59) 2025 | full text | none | **Estimated** "7-10 ms on 80 MHz Cortex-M4", 0.66 mJ from assumed 20 mA x 3.3 V | AE on IoT-23, 47 KB; threshold chosen by best F1 on test labels; FPR 4.8-12% | IoT-23 random 80/20; NS-3 simulation | Shows threshold choice ignored as an operating-point problem; no hardware |
| Dehrouyeh et al., IEEE Access 12 (2024) (arXiv 2404.16894) | full text (case study) | **ESP32** (240 MHz Xtensa), TFLM, PlatformIO, samples sent over UDP | **Measured** latency only (mean over ~28k samples: 246.8 us vs 230.9 us simulated) | 3x8 MLP; device metrics slightly differ from PC ("32-bit vs 64-bit precision") | EVCI dataset, 5-fold CV | Closest real deployment of an MLP IDS. No energy, no bit-exactness; device != PC accepted silently. We can show exact agreement (integer model) |
| Park et al., arXiv 2609.27202 (2026), federated TinyML IDS | full text | none in body; conclusion says "ESP32-class", future work says MCU deployment is still to be done (internal inconsistency) | Latency 1.89 -> 0.48 ms, platform not stated | 14-38 KB TFLite INT8 | CIC-IDS2017, random stratified split | Typical overclaim; cite only for compression trend |
| Rivas Pizarro et al., arXiv 2509.01592 (NPIC&HMIT 2025), radiation IDS | full text | **Intel i7 laptop** only | PC measurements labelled "TinyML" | XGBoost ONNX dynamic quant, 7.4 KB | synthetic dataset | "TinyML" without an MCU |
| Nassef et al., Sci. Rep. 2026 (PMC13269888) | full text | workstation + Raspberry Pi 4 "preliminary" | no MCU; energy not measured | GAT+BiGRU, 8-12 MB | 5-fold CV, PCA/SMOTE possibly before split | Title says TinyML; nothing on MCU |
| Selvaraj et al., JISIS 15(3) 2025 (SMEESI) | full text | **Arduino Nano 33 BLE Sense (nRF52840, Cortex-M4F 64 MHz)**, TFLM | Power/energy "measured" but no instrument named; energy reported per *feature* (205-220 mJ) with competitors' values on the same device | Q-CNN + AE, Edge-IIoTset | random split | Same chip family as ours, but method is not reproducible; numbers implausible (200 mJ/inference). Cite cautiously or not at all |
| De Bernardis et al., Electronics 15(13):2869 (2026) | **abstract only (403)** | two real MCUs (one reportedly ESP32-C3) | measured on device (per abstract) | LR/DT/RF/XGB/LightGBM/MLP, INT8; DT 5.6 us; LightGBM INT8 best trade-off | TON_IoT, 5-fold CV; cross-domain test on CICIoT2023 shows large shift | **Closest prior work** (real MCU + shift to CICIoT2023). Need full PDF |
| Ortiz-Garces et al., Internet of Things 39 (2026), doi 10.1016/j.iot.2026.102034 | **abstract only (403)** | "Cortex-M4" hardware validation (board unknown) | measured (per abstract): 2.48 ms, <20 KB RAM, ~0.39 mJ | NAS-optimised NN | IoT-23 F1 > 0.95 | Second closest (IoT-23, measured energy). Need full PDF |

## B. Measurement methodology sources

| Source | Read | Method | Use for us |
|---|---|---|---|
| MLPerf Tiny (Banbury et al., arXiv 2106.07597) | full text (method) | Latency: on-DUT timer >= 1 kHz, run >= 10 s and >= 10 inferences, 5 repetitions, **median**. Energy: DUT powered by an energy monitor (LPM01A / Joulescope / Keysight), GPIO timestamp marks the window, level-shifter power excluded; uJ/inference, median of 5. Accuracy mode on full validation set; post-training quantisation allowed | Adopt directly: median of 5 runs, GPIO-marked energy window, report accuracy on device |
| Profentzas, Almgren, Landsiedel, CPS-IoTBench 2021 (doi 10.1145/3458473.3458823) | full text | **nRF52840-DK** (Cortex-M4 64 MHz), **Nordic Power Profiler Kit**, cycle-counter timing, 20 repetitions, energy = average current x 3.3 V x time; CMSIS-NN vs TFLM vs uTensor | Same MCU family + same instrument as ours; CMSIS-NN 4-5x lower energy than TFLM; note their PPK capacitor-charging artefact |
| Energy/latency phases on STM32N6 (arXiv 2505.15622) | full text | 50 mOhm shunt + oscilloscope, two GPIO triggers for pre-/inference/post phases, 1000 runs, mean +- sd, EDP | Report pre-processing (feature scaling) separately from inference |
| CMSIS-NN (Lai et al., arXiv 1801.06601) | known | 4.6x speed, 4.9x energy vs baseline kernels on Cortex-M | Optional comparison: plain C vs CMSIS-NN fully-connected s8 |

## C. What this means for the nRF52832 test

Gaps in the literature that our test can close (all verifiable with the existing model):
1. Most "TinyML IDS" papers report estimates or PC/Raspberry Pi timings. A measured nRF52832 number is itself a contribution.
2. No paper verifies that the device gives the same predictions as the PC model. Our integer-only model can be checked for exact agreement on 1,000+ real test flows (Dehrouyeh et al. accept small device/PC differences).
3. No paper measures the cost of the **operating-point repair** on the device: commissioning-quantile threshold (sort of K = 1,000 scores) and the KS check. This links the hardware result to P1's main contribution.
4. Most use random splits (inflated accuracy); ours keeps the strict protocol, so on-device accuracy is the honest holdout number.

Protocol to adopt (MLPerf Tiny + Profentzas):
- Latency: DWT cycle counter, warm-up, >= 1,000 inferences, 5 repetitions, median and p99, single-flow latency and feature-scaling (pre-processing) cost separately.
- Memory: flash/RAM from the linker map.
- Accuracy/agreement: on-device class = PC class on all test vectors; on-device macro-F1 on the included vectors.
- Energy: PPK2 in source-meter mode at 3.0 V, GPIO pin high during the inference loop, energy = integral of current x voltage over the marked window minus idle; else multimeter long-loop average; else datasheet estimate (labelled).
- Optional: CMSIS-NN arm_fully_connected_s8 vs plain C.

Still needed: full PDFs of De Bernardis et al. (Electronics 2026) and Ortiz-Garces et al. (Internet of Things 2026), via browser download.

## D. Update 2026-11-04: board results and comparison with similar papers

Our nRF52832 result (D:\shield_run\mcu\board_results.txt): int8 SHIELD students, 64 MHz Cortex-M4F, measured with the
cycle counter (median of 5 x 1,000 flows): CICIoT2023 34-class 14,690 params / 16.6 KB, 1.71 ms per flow (585/s);
CICIoMT2024 binary 2,978 params / 3.8 KB, 0.41 ms (2,411/s); 0 of 1,000 predictions differ from the PC, Step 1 threshold
and KS distance identical to the PC; Step 1 on 1,000 scores 6 ms, KS 1,000 vs 1,000 10-11 ms. Energy not measured.

New / re-checked papers (this session):
| Paper | Read | Hardware | Measured? | Numbers | Device = PC checked? |
|---|---|---|---|---|---|
| De Bernardis et al., Electronics 15(13):2869, 2026 | abstract (citedrive) | Arduino Mega 2560 (AVR 16 MHz) and ESP32-C3 (RISC-V 160 MHz) | yes, 500 inferences | DT 5.6 us; MLP INT8 13.03 KB; LightGBM INT8 73.85 KB; TON_IoT 5-fold CV F1 0.999; CICIoT2023 shift delta = 5.95 | not stated |
| Ortiz-Garces et al. (NAS tinyML IDS), Internet of Things 2026, S2542660526001642 | abstract (search) | Cortex-M4 (board not given) | yes (cycle counter) | < 2.5 ms, < 20 KB RAM, about 0.39 mJ per inference; IoT-23 | not stated |
| STM32F746 CAN IDS, Springer chapter 2025/26 | snippet | STM32F746 (Cortex-M7 216 MHz) | yes | 8.24 ms, 21.7 KiB RAM, 41.5 KiB flash | not stated |
| ESP32-WROOM-DA anomaly IDS, Springer chapter 2025 | snippet | ESP32 (240 MHz) | yes | mean 3.2 ms (0.23 to 10.5), 8 KB SRAM | not stated |
| Diab et al., ETECOM 2025 (arXiv 2512.02272) | abstract | Raspberry Pi 3 B+ (not an MCU) | yes | LightGBM 75 KB, 95.3% acc; HW-NAS CNN 190 KB | no |
| Tsai, Preprints 202603.0817, 2026 | snippet | STM32N6 (Cortex-M55 + NPU) | latency yes, energy estimated | 0.29 to 0.46 ms INT8 on NPU; 44 to 69 uJ estimated; IoT-23 | not stated |
| Ullah et al., CMC 2026 | full text (earlier) | none (analytical Cortex-M4) | estimated | 0.083 ms, 0.0083 mJ | no |

Position for P1 v5: latency of the same order as the measured MCU IDS papers (0.4 to 1.7 ms on a 64 MHz M4F, against
2.5 ms on an M4, 3.2 ms on a 240 MHz ESP32, 8.2 ms on a 216 MHz M7); what is new is (1) exact device = PC agreement on
predictions, threshold and KS value, and (2) the measured on-device cost of the operating-point repair (Step 1, KS).
No paper found reports either. Do not claim the fastest; the speeds are not comparable across models, clocks and
timing scopes.
