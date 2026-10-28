"""Streaming download with size / checksum verification (for the small audio corpora).

``download.fetch`` reads a whole response into memory and cannot verify it; the audio corpora
added for the sound-event model are pinned by size and md5/sha256 instead, and a file that does
not match is deleted rather than kept. Nothing here writes outside the destination directory.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from homeease.core.util import get_logger

log = get_logger(__name__)
UA = "homeease-research/0.1 (academic project)"


def _request(url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> urllib.request.Request:
    return urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})


def download_checked(
    url: str,
    dest: Path,
    *,
    size: int | None = None,
    md5: str | None = None,
    sha256: str | None = None,
    retries: int = 3,
    timeout: float = 120.0,
    max_bytes: int = 150 * 1024 * 1024,
) -> dict[str, Any]:
    """Download ``url`` to ``dest`` (atomically, via ``.part``) and verify the pinned values.

    Returns a log entry ``{url, path, status, bytes, sha256, md5}``; ``status`` is ``downloaded``,
    ``cached`` (file present and matching) or ``failed``. A present file that does not match the pin
    is re-downloaded; a downloaded file that does not match is removed and reported as failed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        got = _hashes(dest)
        if _matches(got, dest.stat().st_size, size, md5, sha256):
            return {"url": url, "path": str(dest), "status": "cached", "bytes": dest.stat().st_size, **got}
        log.warning("%s does not match its pin; downloading again", dest.name)
    last: Exception | str | None = None
    for attempt in range(retries):
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            h5, h2, n = hashlib.md5(), hashlib.sha256(), 0
            with urllib.request.urlopen(_request(url), timeout=timeout) as r, open(part, "wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    n += len(b)
                    if n > max_bytes:
                        raise ValueError(f"response larger than the {max_bytes} byte limit")
                    h5.update(b)
                    h2.update(b)
                    fh.write(b)
            got = {"md5": h5.hexdigest(), "sha256": h2.hexdigest()}
            if not _matches(got, n, size, md5, sha256):
                part.unlink(missing_ok=True)
                last = f"checksum/size mismatch (got {n} bytes, md5 {got['md5']}, sha256 {got['sha256'][:12]})"
                continue
            part.replace(dest)
            return {"url": url, "path": str(dest), "status": "downloaded", "bytes": n, **got}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            part.unlink(missing_ok=True)
            last = repr(e)
            time.sleep(1.5 * (attempt + 1))
    return {"url": url, "path": str(dest), "status": "failed", "error": str(last)}


def _hashes(p: Path) -> dict[str, str]:
    h5, h2 = hashlib.md5(), hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h5.update(b)
            h2.update(b)
    return {"md5": h5.hexdigest(), "sha256": h2.hexdigest()}


def _matches(got: dict[str, str], n: int, size: int | None, md5: str | None, sha256: str | None) -> bool:
    return (size is None or n == size) and (md5 is None or got["md5"] == md5) and (sha256 is None or got["sha256"] == sha256)


def get_json(url: str, *, data: dict[str, Any] | None = None, timeout: float = 60.0, retries: int = 3) -> Any:
    """GET (or POST when ``data`` is given) a small JSON document."""
    body = json.dumps(data).encode() if data is not None else None
    hdr = {"Content-Type": "application/json"} if data is not None else {}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_request(url, body, hdr), timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last!r}")


def total_bytes(entries: Iterable[dict[str, Any]]) -> int:
    """Bytes actually transferred (cached files cost nothing)."""
    return sum(int(e.get("bytes", 0)) for e in entries if e.get("status") == "downloaded")
