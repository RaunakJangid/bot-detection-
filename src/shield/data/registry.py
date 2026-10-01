"""Dataset registry (configs/datasets.yaml): which processed datasets exist, their source, protocol,
tasks, seeds and KD variants, and which optional stages run on them."""

from __future__ import annotations

from shield.utils.config import load_config

SOURCES = ("ciciot", "ciciomt", "botiot")


class Registry:
    def __init__(self, smoke: bool = False):
        cfg = load_config("datasets", smoke)
        self.specs: dict[str, dict] = cfg["datasets"]
        self.leaky: dict[str, list[str]] = cfg["leaky_features"]
        self.stages: dict[str, list[str]] = cfg["stages"]

    @property
    def names(self) -> list[str]:
        return list(self.specs)

    def spec(self, ds: str) -> dict:
        if ds not in self.specs:
            raise KeyError(f"Unknown dataset {ds!r}; known: {self.names}")
        return self.specs[ds]

    def source(self, ds: str) -> str:
        return self.spec(ds)["source"]

    def dropped(self, ds: str) -> list[str]:
        s = self.spec(ds)
        return list(self.leaky.get(s["source"], [])) if s["protocol"] == "strict" else []

    def resolve(self, arg: str | None) -> list[str]:
        """'all', a group name (main / cross / standard), a dataset name, or a comma-separated mix."""
        if arg in (None, "all"):
            return self.names
        out: list[str] = []
        for part in arg.split(","):
            part = part.strip()
            group = [n for n, s in self.specs.items() if s.get("group") == part]
            if not group:
                self.spec(part)  # raises for unknown names
                group = [part]
            out += [n for n in group if n not in out]
        return out

    def for_stage(self, stage: str, names: list[str]) -> list[str]:
        groups = set(self.stages.get(stage, []))
        return [n for n in names if self.spec(n).get("group") in groups]

    def tasks(self, ds: str, only: str | None = None) -> list[str]:
        return [t for t in self.spec(ds)["tasks"] if only is None or t == only]

    def seeds(self, ds: str, default: list[int]) -> list[int]:
        return list(self.spec(ds).get("seeds", default))

    def variants(self, ds: str, default: list[str]) -> list[str]:
        return list(self.spec(ds).get("variants", default))
