from transformers import AutoTokenizer
import sentencepiece as spm
import json
import os
from tqdm import tqdm


# Base model path and dataset path can be configured via env vars
BASE_MODEL_PATH = os.environ.get("METLLM_BASE_MODEL", "/Your/ChatGLM2/MODEL_PATH/")
DATA_PATH = os.environ.get("METLLM_TOKENIZER_DATA", "/Your/ORIGINAL/DATA_PATH/")
OUTPUT_PREFIX = os.environ.get("METLLM_TOKENIZER_PREFIX", "traffic_tokenizer")
VOCAB_SIZE = int(os.environ.get("METLLM_TOKENIZER_VOCAB", "64000"))


def build_training_data():
    write_path = f"{OUTPUT_PREFIX}.txt"
    dataset = []
    with open(DATA_PATH, "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            obj = json.loads(line)
            ins = obj.get("instruction", "")
            out = obj.get("output", "")
            if ins:
                dataset.append(ins)
            if out:
                dataset.append(out)

    with open(write_path, "w", encoding="utf-8") as fout:
        for data in dataset:
            fout.write(data + "\n")
    return write_path


def train_corpus(corpus_path: str):
    # inject domain special symbols
    user_syms = [
        "[CLS]", "[HEAD]", "[BODY]", "[MASK]", "[NOISE]",
        "ip.", "tcp.", "udp.", "eth.", "frame.", "0x",
    ]
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=OUTPUT_PREFIX,
        vocab_size=VOCAB_SIZE,
        user_defined_symbols=user_syms,
        character_coverage=1.0,
        model_type="bpe",
    )


def tokenizer_comparing():
    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    print(base_tok.all_special_tokens)
    print(base_tok.all_special_ids)
    print(base_tok.special_tokens_map)
    with open(f"{OUTPUT_PREFIX}.txt", "r", encoding="utf-8") as fin:
        dataset = fin.readlines()
    count = 0
    len_base = 0
    len_traffic = 0
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(f"./{OUTPUT_PREFIX}.model")
    for text in tqdm(dataset[:100]):
        if len(text) < 10:
            continue
        count += 1
        len_base += len(base_tok.tokenize(text))
        len_traffic += len(sp_model.EncodeAsPieces(text))

    print("base token len", len_base / count)
    print("traffic token len", len_traffic / count)


if __name__ == "__main__":
    corpus = build_training_data()
    train_corpus(corpus)
    # tokenizer_comparing()