"""Download helpers with resume-by-skip, size checks and a provenance log."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from homeease.core.util import sha256_file


def fetch(url: str, dest: Path, retries: int = 3, timeout: float = 120.0) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return {"url": url, "path": str(dest), "status": "cached", "bytes": dest.stat().st_size}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "homeease-research/0.1 (academic project)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return {"url": url, "path": str(dest), "status": "downloaded", "bytes": len(data)}
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    return {"url": url, "path": str(dest), "status": "failed", "error": repr(last)}


def write_download_log(root: Path, entries: list[dict[str, Any]], meta: dict[str, Any]) -> Path:
    log = {"meta": meta, "accessed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": entries}
    for e in entries:
        p = Path(e["path"])
        if e.get("status") in ("downloaded", "cached") and p.is_file() and p.stat().st_size < 50 * 1024 * 1024:
            e["sha256"] = sha256_file(p)
    out = root / "DOWNLOAD_LOG.json"
    out.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return out
