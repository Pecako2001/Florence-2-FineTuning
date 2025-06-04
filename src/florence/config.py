import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


def load_config(path: str) -> Dict[str, Any]:
    """Load a configuration file in JSON or YAML format."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    with p.open("r") as f:
        if p.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ImportError("PyYAML is required for YAML config files")
            return yaml.safe_load(f)
        return json.load(f)
