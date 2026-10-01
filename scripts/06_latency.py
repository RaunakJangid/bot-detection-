"""Inference cost of teacher and students -> outputs/latency/latency.json.

CPU numbers are single-core (pinned, 1 thread): the constrained-edge proxy used for controller capacity.
"""

import torch

from shield.cli import parser, paths_for, selected_datasets, tasks_for
from shield.eval.latency import cpu_latency, flops_per_sample, gpu_throughput, model_size_kb
from shield.kd.trainer import kd_dir
from shield.models.student import StudentMLP, count_params
from shield.models.train import load_teacher, teacher_dir
from shield.utils.config import load_config
from shield.utils.device import hardware_record
from shield.utils.io import load_json, save_json
from shield.utils.logging import get_logger

log = get_logger("latency")


def profile(model, n_features, runs, warmup, flops_model=None) -> dict:
    rec = {"n_features": n_features, "size_kb": round(model_size_kb(model), 2)}
    rec["flops"] = flops_per_sample(flops_model or model, n_features)
    rec.update(cpu_latency(model, n_features, runs=runs, warmup=warmup))
    return rec


def main():
    args = parser(__doc__).parse_args()
    paths = paths_for(args)
    tcfg, kcfg = load_config("teacher", args.smoke), load_config("kd", args.smoke)
    runs = 50 if args.smoke else 1000
    out_file = paths.out("latency") / "latency.json"
    table = load_json(out_file) if out_file.exists() and not args.force else {}
    for ds in selected_datasets(args):
        for task in tasks_for(tcfg["tasks"], ds, args.task):
            key = f"{ds}_{task}"
            entry = {}
            tdir = teacher_dir(paths, ds, task, tcfg["model"], kcfg["teacher_seed"])
            teacher, ckpt = load_teacher(tdir, torch.device("cpu"))
            entry["teacher"] = profile(teacher, ckpt["n_features"], max(20, runs // 5), 10)
            entry["teacher"]["params"] = count_params(teacher)
            entry["teacher"]["gpu_throughput"] = gpu_throughput(teacher, ckpt["n_features"])
            entry["teacher"]["macro_f1"] = load_json(tdir / "metrics.json")["test"]["macro_f1"]
            seed = kcfg["seeds"][0]
            for variant in kcfg["variants"]:
                sdir = kd_dir(paths, ds, task, variant, seed)
                if not (sdir / "student.pt").exists():
                    log.warning("missing %s, skipping", sdir)
                    continue
                ck = torch.load(sdir / "student.pt", map_location="cpu", weights_only=False)
                student = StudentMLP(len(ck["selected"]), ck["n_classes"], ck["hidden"])
                student.load_state_dict(ck["state_dict"])
                metrics = load_json(sdir / "metrics.json")
                entry[variant] = profile(student, len(ck["selected"]), runs, 100)
                entry[variant].update(params=count_params(student), macro_f1=metrics["test"]["macro_f1"])
                if variant == "shield":
                    q = torch.load(sdir / "student_int8.pt", map_location="cpu", weights_only=False)
                    entry["shield_int8"] = profile(q, len(ck["selected"]), runs, 100, flops_model=student)
                    entry["shield_int8"].update(params=count_params(student),
                                                macro_f1=metrics["test_int8"]["macro_f1"])
            table[key] = entry
            log.info("%s: teacher %.0f flows/s, shield %s flows/s (single core, batch 256)", key,
                     entry["teacher"]["throughput_b256"],
                     f"{entry['shield']['throughput_b256']:.0f}" if "shield" in entry else "n/a")
    table["_hardware"] = hardware_record()
    save_json(table, out_file)


if __name__ == "__main__":
    main()
