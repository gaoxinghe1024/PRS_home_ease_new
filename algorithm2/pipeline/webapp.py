"""Export the HTTP schema the integrated system talks to."""
import json

from homeease.api import app

from .common import ROOT


def export_openapi():
    (ROOT / "docs/openapi.json").write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    print("Exported docs/openapi.json")
