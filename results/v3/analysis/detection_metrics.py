"""Detection metrics of the final models (ensemble + validation rule): macro precision/recall/F1, accuracy, MCC,
benign false-positive rate (benign flows flagged as any attack) and attack detection rate (attack flows not
called benign)."""
import json

import numpy as np

V3 = r"D:\shield_run\outputs_v3\detector_v4"
for k in ["ciciot_full_class34", "ciciot_full_family", "ciciot_full_binary", "ciciomt_category", "ciciomt_attack",
          "ciciomt_binary", "botiot_category", "botiot_binary"]:
    j = json.load(open(f"{V3}/{k}.json"))
    m = j["test_ens4_rule"]
    pc = m["per_class"]
    ben = next(c for c in pc if c in ("BenignTraffic", "Benign", "Normal"))
    P = np.mean([v["precision"] for v in pc.values()])
    R = np.mean([v["recall"] for v in pc.values()])
    # benign row of the confusion is not stored, so: FPR = 1 - benign recall (benign flows predicted as any attack)
    fpr = 1 - pc[ben]["recall"]
    att = [v for c, v in pc.items() if c != ben]
    print(f"{k}: macroP {100 * P:.2f} macroR {100 * R:.2f} macroF1 {100 * m['macro_f1']:.2f} acc {100 * m['accuracy']:.2f} "
          f"MCC {100 * m['mcc']:.2f} benign-FPR {100 * fpr:.2f} benign-precision {100 * pc[ben]['precision']:.2f}")
