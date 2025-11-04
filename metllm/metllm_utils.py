import os
import json
import random
from typing import Tuple, List


DEFAULT_CONFIG = {
    "mask_prob_header": 0.1,
    "mask_prob_payload": 0.15,
    "mask_token": "[MASK]",
    "cls_token": "[CLS]",
    "head_token": "[HEAD]",
    "body_token": "[BODY]",
}


HEADER_HINT_KEYS = (
    "frame.",
    "eth.",
    "ip.",
    "tcp.",
    "udp.",
    "data.len",
)


def load_metllm_config(config_path: str = "metllm_config.json") -> dict:
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return {**DEFAULT_CONFIG, **cfg}
        except Exception:
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG


def split_header_payload(packet_text: str) -> Tuple[str, str]:
    """
    Heuristic split of a single-line packet description into header-like fields and payload-like content.
    Input example comes from tshark export used in the repo: "field1: v1, field2: v2, tcp.payload: abcd..."
    """
    if not packet_text:
        return "", ""
    parts = [p.strip() for p in packet_text.split(",")]
    header_fields: List[str] = []
    payload_fields: List[str] = []
    for item in parts:
        if not item:
            continue
        # detect key: value
        if ":" in item:
            k, v = item.split(":", 1)
            k = k.strip()
            v = v.strip()
        else:
            k, v = item, ""
        # payload hint
        if k.lower().endswith("payload") or k.lower().endswith("payload_hex") or k.lower() == "tcp.payload":
            payload_fields.append(f"{k}: {v}" if v else k)
        else:
            # header-like
            if any(k.startswith(prefix) for prefix in HEADER_HINT_KEYS):
                header_fields.append(f"{k}: {v}" if v else k)
            else:
                # default to header if unsure
                header_fields.append(f"{k}: {v}" if v else k)
    return ", ".join(header_fields), ", ".join(payload_fields)


def dynamic_mask(text: str, mask_prob: float, mask_token: str) -> str:
    if not text or mask_prob <= 0.0:
        return text
    tokens = text.split(" ")
    masked: List[str] = []
    for tok in tokens:
        if tok and random.random() < mask_prob:
            masked.append(mask_token)
        else:
            masked.append(tok)
    return " ".join(masked)


def build_structured_sequence(header: str, payload: str, cfg: dict | None = None) -> str:
    cfg = cfg or DEFAULT_CONFIG
    cls_t = cfg.get("cls_token", "[CLS]")
    head_t = cfg.get("head_token", "[HEAD]")
    body_t = cfg.get("body_token", "[BODY]")
    header = header or ""
    payload = payload or ""
    return f"{cls_t} {header} {head_t} {payload} {body_t}"


def apply_masking_to_segments(header: str, payload: str, cfg: dict | None = None) -> Tuple[str, str]:
    cfg = cfg or DEFAULT_CONFIG
    ph = float(cfg.get("mask_prob_header", 0.1))
    pp = float(cfg.get("mask_prob_payload", 0.15))
    mk = cfg.get("mask_token", "[MASK]")
    return dynamic_mask(header, ph, mk), dynamic_mask(payload, pp, mk)


