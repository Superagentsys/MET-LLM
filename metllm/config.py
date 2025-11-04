import json
import os
from typing import Any, Dict


def load_json_config(config_path: str = "config.json") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # 环境变量覆盖
    model_path = os.environ.get("METLLM_MODEL_PATH")
    if model_path:
        cfg["model_path"] = model_path
    return cfg


