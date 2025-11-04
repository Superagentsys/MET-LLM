from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from metllm.core.embedding import TrafficEmbeddingProcessor, EmbeddingConfig


@dataclass
class TrafficEmbedderConfig:
    mask_prob_header: float | None = None
    mask_prob_payload: float | None = None
    mask_token: str | None = None
    cls_token: str | None = None
    head_token: str | None = None
    body_token: str | None = None


class TrafficEmbedder:
    """新命名封装，向后兼容旧实现。"""

    def __init__(self, config: TrafficEmbedderConfig | None = None) -> None:
        base_cfg = EmbeddingConfig.load()
        if config is not None:
            if config.mask_prob_header is not None:
                base_cfg.mask_prob_header = float(config.mask_prob_header)
            if config.mask_prob_payload is not None:
                base_cfg.mask_prob_payload = float(config.mask_prob_payload)
            if config.mask_token is not None:
                base_cfg.mask_token = str(config.mask_token)
            if config.cls_token is not None:
                base_cfg.cls_token = str(config.cls_token)
            if config.head_token is not None:
                base_cfg.head_token = str(config.head_token)
            if config.body_token is not None:
                base_cfg.body_token = str(config.body_token)
        self._impl = TrafficEmbeddingProcessor(base_cfg)

    def split(self, packet_text: str) -> Tuple[str, str]:
        return self._impl.split(packet_text)

    def to_structured(self, header: str, payload: str) -> str:
        return self._impl.to_structured(header, payload)

    def to_structured_with_mask(self, header: str, payload: str) -> str:
        return self._impl.to_structured_with_mask(header, payload)


