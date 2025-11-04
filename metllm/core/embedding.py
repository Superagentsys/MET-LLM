from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from metllm.utils import (
    load_metllm_config,
    split_header_payload,
    apply_masking_to_segments,
    build_structured_sequence,
)


@dataclass
class EmbeddingConfig:
    mask_prob_header: float
    mask_prob_payload: float
    mask_token: str
    cls_token: str
    head_token: str
    body_token: str

    @staticmethod
    def load(config_path: str = "metllm_config.json") -> "EmbeddingConfig":
        cfg = load_metllm_config(config_path)
        return EmbeddingConfig(
            mask_prob_header=float(cfg.get("mask_prob_header", 0.1)),
            mask_prob_payload=float(cfg.get("mask_prob_payload", 0.15)),
            mask_token=str(cfg.get("mask_token", "[MASK]")),
            cls_token=str(cfg.get("cls_token", "[CLS]")),
            head_token=str(cfg.get("head_token", "[HEAD]")),
            body_token=str(cfg.get("body_token", "[BODY]")),
        )


class TrafficEmbeddingProcessor:
    """将原始抓包文本转换为 LLM 可用的结构化序列，并支持训练期动态掩码。"""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._cfg = config or EmbeddingConfig.load()

    def split(self, packet_text: str) -> Tuple[str, str]:
        return split_header_payload(packet_text)

    def to_structured(self, header: str, payload: str) -> str:
        return build_structured_sequence(header, payload, {
            "cls_token": self._cfg.cls_token,
            "head_token": self._cfg.head_token,
            "body_token": self._cfg.body_token,
        })

    def to_structured_with_mask(self, header: str, payload: str) -> str:
        h_m, p_m = apply_masking_to_segments(header, payload, {
            "mask_prob_header": self._cfg.mask_prob_header,
            "mask_prob_payload": self._cfg.mask_prob_payload,
            "mask_token": self._cfg.mask_token,
            "cls_token": self._cfg.cls_token,
            "head_token": self._cfg.head_token,
            "body_token": self._cfg.body_token,
        })
        return self.to_structured(h_m, p_m)


