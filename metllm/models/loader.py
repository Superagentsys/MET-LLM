from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
import torch


def load_tokenizer_and_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        model_config = None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config, trust_remote_code=True)
    except Exception:
        model = AutoModel.from_pretrained(model_path, config=model_config, trust_remote_code=True)
    return tokenizer, model


def metllm_llm(model, tokenizer, text: str, max_new_tokens: int = 256, top_p: float = 0.8, temperature: float = 0.8) -> Tuple[str, list]:
    if hasattr(model, "chat"):
        try:
            response, history = model.chat(tokenizer, text, history=[])
            return response, history
        except Exception:
            pass
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
        )
    gen_ids = outputs[0]
    prompt_len = inputs["input_ids"].shape[1]
    gen_text = tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=True)
    return gen_text, []


