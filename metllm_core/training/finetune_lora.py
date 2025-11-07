from __future__ import annotations

import json
from typing import Any, Dict

import fire
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def _format_example(example: Dict[str, Any], tokenizer: AutoTokenizer, max_source_len: int, max_target_len: int) -> Dict[str, Any]:
    if "messages" in example:
        try:
            ids = tokenizer.apply_chat_template(example["messages"], tokenize=True, return_tensors=None)
            return {"input_ids": ids[: max_source_len + max_target_len]}
        except Exception:
            pass
    instr = example.get("instruction", None)
    out = example.get("output", None)
    if instr is not None and out is not None:
        prompt = str(instr)
        target = str(out)
        text = prompt + "\n" + target
    else:
        text = example.get("text", "")
    ids = tokenizer(text, truncation=True, max_length=max_source_len + max_target_len)["input_ids"]
    return {"input_ids": ids}


def main(
    model_path: str,
    train_file: str,
    output_dir: str = "./output",
    lr: float = 5e-5,
    max_steps: int = 3000,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    bf16: bool = True,
    max_source_length: int = 1024,
    max_target_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16 if bf16 else torch.float32)

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": train_file})
    def proc(ex):
        return _format_example(ex, tokenizer, max_source_length, max_target_length)
    train_ds = dataset["train"].map(proc, remove_columns=dataset["train"].column_names)

    def collate_fn(features):
        ids = [f["input_ids"] for f in features]
        batch = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        batch["labels"] = labels
        return batch

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=bf16,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    fire.Fire(main)


