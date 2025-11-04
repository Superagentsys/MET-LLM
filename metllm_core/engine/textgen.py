from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Any

from metllm.core.inference_engine import LLMInferenceEngine, InferenceConfig


@dataclass
class TextGenConfig:
    model_path: str
    top_p: float = 0.8
    temperature: float = 0.8
    max_new_tokens: int = 256


class TextGenEngine:
    """新命名封装，向后兼容旧推理引擎。"""

    def __init__(self, config: TextGenConfig) -> None:
        self._impl = LLMInferenceEngine(
            InferenceConfig(
                model_path=config.model_path,
                top_p=config.top_p,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
            )
        )

    @property
    def tokenizer(self):
        return self._impl.tokenizer

    @property
    def model(self):
        return self._impl.model

    def chat(self, text: str) -> Tuple[str, Any]:
        return self._impl.chat(text)


