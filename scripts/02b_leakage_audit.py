"""Leakage audit: how well does each feature classify on its own?

Runs on the standard-protocol datasets (all original features, including the ones the strict protocol
removes) at their finest task. A single feature that separates many classes almost perfectly is a
shortcut suspect (capture time, record order, host identity) rather than traffic behaviour.
Output: outputs/leakage/<dataset>_<task>.csv and a combined summary.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from shield.cli import paths_for, parser, registry, selected_datasets
from shield.data.datasets import load_processed, stratified_sample
from shield.utils.config import load_config
from shield.utils.io import save_json
from shield.utils.logging import get_logger

log = get_logger("leakage")


def audit(data, cfg: dict, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tr, te = data.splits["train"], data.splits["test"]
    i_tr = stratified_sample(tr.y, min(cfg["train_rows"], len(tr.y)), rng, min_per_class=20)
    i_te = stratified_sample(te.y, min(cfg["test_rows"], len(te.y)), rng, min_per_class=20)
    Xtr, ytr, Xte, yte = np.asarray(tr.X[i_tr]), tr.y[i_tr], np.asarray(te.X[i_te]), te.y[i_te]
    labels = np.arange(data.n_classes)

    def score(cols):
        clf = DecisionTreeClassifier(max_depth=cfg["max_depth"], random_state=seed).fit(Xtr[:, cols], ytr)
        return f1_score(yte, clf.predict(Xte[:, cols]), labels=labels, average="macro", zero_division=0)

    full = score(list(range(data.n_features)))
    rows = [{"feature": f, "macro_f1_alone": score([j])} for j, f in enumerate(data.features)]
    df = pd.DataFrame(rows).sort_values("macro_f1_alone", ascending=False).reset_index(drop=True)
    df["share_of_all_features"] = df["macro_f1_alone"] / max(full, 1e-12)
    df["chance_macro_f1"] = 1.0 / data.n_classes
    df.attrs["full"] = full
    return df


def main():
    args = parser(__doc__).parse_args()
    if args.dataset == "all":
        args.dataset = "standard"
    paths, cfg, reg = paths_for(args), load_config("data", args.smoke)["audit"], registry(args)
    out_dir = paths.out("leakage")
    summary = {}
    for ds in selected_datasets(args):
        task = reg.tasks(ds)[-1]  # finest granularity listed last
        if not (paths.processed(ds) / "meta.json").exists():
            log.warning("%s not processed, skipping", ds)
            continue
        data = load_processed(paths.processed(ds), task, splits=("train", "test"))
        df = audit(data, cfg)
        df.to_csv(out_dir / f"{ds}_{task}.csv", index=False)
        leaky = set(reg.leaky.get(reg.source(ds), []))
        summary[f"{ds}_{task}"] = {
            "all_features_macro_f1": df.attrs["full"], "chance": 1.0 / data.n_classes,
            "top5": df.head(5)[["feature", "macro_f1_alone"]].to_dict("records"),
            "removed_by_strict_protocol": df[df.feature.isin(leaky)][["feature", "macro_f1_alone"]].to_dict("records"),
        }
        log.info("%s/%s: all features %.3f | top single features:\n%s", ds, task, df.attrs["full"],
                 df.head(8).to_string(index=False))
    save_json(summary, out_dir / "summary.json")


if __name__ == "__main__":
    main()
