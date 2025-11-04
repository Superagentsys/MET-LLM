import fire
from tokenization import metllm_tokenizer as mtok


def build():
    return mtok.build_training_data()


def train(corpus_path: str | None = None):
    path = corpus_path or mtok.build_training_data()
    return mtok.train_corpus(path)


def compare():
    return mtok.tokenizer_comparing()


if __name__ == "__main__":
    fire.Fire({
        "build": build,
        "train": train,
        "compare": compare,
    })


