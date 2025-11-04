from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Any

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig


@dataclass
class InferenceConfig:
    model_path: str
    top_p: float = 0.8
    temperature: float = 0.8
    max_new_tokens: int = 256


class LLMInferenceEngine:
    """统一封装 HF CausalLM/Chat 接口以便 DeepSeek/LLAMA/GLM 兼容。"""

    def __init__(self, config: InferenceConfig) -> None:
        self._cfg = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
        try:
            model_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
        except Exception:
            model_config = None
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path, config=model_config, trust_remote_code=True
            )
        except Exception:
            self.model = AutoModel.from_pretrained(
                config.model_path, config=model_config, trust_remote_code=True
            )
        self.model = self.model.eval()

    def chat(self, text: str) -> Tuple[str, Any]:
        if hasattr(self.model, "chat"):
            try:
                response, history = self.model.chat(self.tokenizer, text, history=[])
                return response, history
            except Exception:
                pass
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                top_p=self._cfg.top_p,
                temperature=self._cfg.temperature,
                max_new_tokens=self._cfg.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id,
            )
        gen_ids = outputs[0]
        prompt_len = inputs["input_ids"].shape[1]
        gen_text = self.tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=True)
        return gen_text, []


