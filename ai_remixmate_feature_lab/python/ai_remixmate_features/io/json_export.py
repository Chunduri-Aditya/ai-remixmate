from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_json(payload: dict[str, Any], output_path: str) -> dict[str, Any]:
    """Write a JSON payload to an explicit path and return export metadata."""
    path = Path(output_path)
    if not path.parent.exists():
        raise FileNotFoundError(f"parent directory does not exist: {path.parent}")
    path.write_text(json.dumps(payload, indent=2) + "
", encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size}
