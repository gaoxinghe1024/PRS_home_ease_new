"""Experiment run bookkeeping: provenance, artifacts and honest status values.

status values:
  completed      metrics computed on the stated data
  synthetic_only only synthetic data were available: software validation, not accuracy
  data_pending   the required real data are not available (what is missing is listed)
  not_evaluated  the experiment could not run for another stated reason
  failed         an error occurred (traceback saved)
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from homeease.core.paths import artifacts_dir
from homeease.core.util import code_version, dumps

STATUSES = {"completed", "synthetic_only", "data_pending", "not_evaluated", "failed", "running"}


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "scipy", "scikit-learn", "torch", "mediapipe", "opencv-python", "onnxruntime", "rtmlib", "faster-whisper", "silero-vad"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


class Run:
    def __init__(self, suite: str, name: str, config: dict[str, Any], data_source: str, seed: int | None = None, split_hash: str | None = None) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]
        self.run_id = f"{stamp}_{suite}_{name}_{cfg_hash}"
        self.dir = artifacts_dir() / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.suite, self.name, self.config, self.data_source = suite, name, config, data_source
        self.provenance = {
            "run_id": self.run_id, "suite": suite, "name": name, "config_hash": cfg_hash, "code_version": code_version(),
            "split_hash": split_hash, "seed": seed, "data_source": data_source, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "versions": _versions(), "device": platform.machine(),
        }
        (self.dir / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(config, default=str)), sort_keys=False, allow_unicode=True), encoding="utf-8")
        self._t0 = time.time()

    def predictions(self, rows: list[dict[str, Any]]) -> None:
        with open(self.dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(dumps(r) + "\n")

    def summary_csv(self, rows: list[dict[str, Any]], name: str = "summary.csv") -> None:
        if not rows:
            return
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(self.dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})

    def figure(self, fig: Any, name: str) -> str:
        p = self.dir / f"{name}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        return p.name

    def finish(self, status: str, metrics: dict[str, Any], headline: str = "", report_md: str = "", missing: list[str] | None = None) -> dict[str, Any]:
        assert status in STATUSES
        out = {
            "status": status, "suite": self.suite, "name": self.name, "data_source": self.data_source, "headline": headline,
            "missing": missing or [], "metrics": metrics, "provenance": {**self.provenance, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - self._t0, 1)},
        }
        (self.dir / "metrics.json").write_text(dumps(out, indent=2), encoding="utf-8")
        md = f"# {self.suite} / {self.name}\n\n- run_id: `{self.run_id}`\n- status: **{status}**\n- data: {self.data_source}\n- code: `{self.provenance['code_version']}` · config `{self.provenance['config_hash']}` · split `{self.provenance['split_hash']}` · seed `{self.provenance['seed']}`\n"
        if missing:
            md += "\n## Missing inputs\n" + "\n".join(f"- {m}" for m in missing) + "\n"
        if headline:
            md += f"\n## Headline\n{headline}\n"
        md += "\n" + (report_md or "")
        (self.dir / "report.md").write_text(md, encoding="utf-8")
        return out

    def fail(self, err: BaseException) -> dict[str, Any]:
        (self.dir / "traceback.txt").write_text("".join(traceback.format_exception(err)), encoding="utf-8")
        return self.finish("failed", {"error": repr(err)}, headline=f"failed: {err!r}")


def fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return f"[{v[0]:.{nd}f}, {v[1]:.{nd}f}]"
    return str(v)


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "_no rows_\n"
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n"
    return out


def write_suite_index(suite: str, entries: list[dict[str, Any]]) -> Path:
    d = artifacts_dir() / f"_suite_{suite}_{time.strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    md = f"# Suite `{suite}` summary\n\nEach row links to a run directory with config, predictions, metrics and report.\n\n"
    md += md_table(entries, ["id", "experiment", "status", "data", "headline", "run_id"])
    (d / "report.md").write_text(md, encoding="utf-8")
    (d / "metrics.json").write_text(dumps({"status": "completed", "suite": suite, "name": "suite_index", "entries": entries}, indent=2), encoding="utf-8")
    return d
