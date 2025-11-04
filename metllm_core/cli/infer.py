import json
import fire

from metllm.core.embedding import TrafficEmbeddingProcessor
from metllm.core.inference_engine import LLMInferenceEngine, InferenceConfig


def main(config: str, prompt: str):
    with open(config, "r", encoding="utf-8") as fin:
        cfg = json.load(fin)
    engine = LLMInferenceEngine(InferenceConfig(model_path=cfg["model_path"]))
    embedder = TrafficEmbeddingProcessor()

    structured = prompt
    if "<packet>" in prompt:
        pkt = "<packet>".join(prompt.split("<packet>")[1:])
        h, p = embedder.split(pkt)
        structured = embedder.to_structured(h, p)
    out, _ = engine.chat(structured)
    print(out)


if __name__ == "__main__":
    fire.Fire(main)


