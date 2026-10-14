import yaml
from pathlib import Path
from typing import Any, Dict, Optional

_config: Optional[Dict] = None
_config_path = Path("config.yaml")


def load_config(path: str = "config.yaml") -> Dict:
    global _config, _config_path
    _config_path = Path(path)
    if _config_path.exists():
        with open(_config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f) or {}
    else:
        _config = {}
    return _config


def get_config() -> Dict:
    global _config
    if _config is None:
        load_config()
    return _config


def get_threshold(name: str, default: float = 0.5) -> float:
    return get_config().get("thresholds", {}).get(name, default)


def get_retrieval(name: str, default: Any = None) -> Any:
    return get_config().get("retrieval", {}).get(name, default)


def get_pipeline(name: str, default: Any = None) -> Any:
    return get_config().get("pipeline", {}).get(name, default)


def get_model(name: str, default: Any = None) -> Any:
    return get_config().get("models", {}).get(name, default)


def get_nli(name: str, default: Any = None) -> Any:
    return get_config().get("nli", {}).get(name, default)


load_config()