from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
import fire
import torch
import json
import os
from metllm.core.embedding import TrafficEmbeddingProcessor
from metllm.core.inference_engine import LLMInferenceEngine, InferenceConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_model(model, ptuning_path):
    # ChatGLM 兼容：若存在 prefix encoder 权重则加载；DeepSeek 不会有此前缀，忽略
    if ptuning_path is not None and os.path.exists(os.path.join(ptuning_path, "pytorch_model.bin")):
        try:
            prefix_state_dict = torch.load(
                os.path.join(ptuning_path, "pytorch_model.bin"))
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            if hasattr(model, "transformer") and hasattr(model.transformer, "prefix_encoder"):
                model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
                model = model.half().cuda()
                model.transformer.prefix_encoder.float()
        except Exception:
            pass
    return model


def prompt_processing(prompt):
    instruction_text = prompt.split("<packet>")[0]
    traffic_data = "<packet>" + "<packet>".join(prompt.split("<packet>")[1:])

    return instruction_text, traffic_data


def preprompt(task, traffic_data):
    """Preprompts in LLMs for downstream traffic pattern learning"""
    prepromt_set = {
        "MTD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and "
               "payloads. Please conduct the ENCRYPTED MALWARE DETECTION TASK to determine which application "
               "category the encrypted beign or malicious traffic belongs to. The categories include 'BitTorrent, "
               "FTP, Facetime, Gmail, MySQL, Outlook, SMB, Skype, Weibo, WorldOfWarcraft,Cridex, Geodo, Htbot, Miuref, "
               "Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus'.\n",
        "BND": "Given the following traffic data <packet> that contains protocol fields, traffic features, "
               "and payloads. Please conduct the BOTNET DETECTION TASK to determine which type of network the "
               "traffic belongs to. The categories include 'IRC, Neris, RBot, Virut, normal'.\n",
        "WAD": "Classify the given HTTP request into benign and malicious categories. Each HTTP request will consist "
               "of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an "
               "HTTP request, please output an 'exception'. Only output 'malicious' or 'benign', no additional output "
               "is required. The given HTTP request is as follows:\n",
        "AAD": "Classify the given HTTP request into normal and abnormal categories. Each HTTP request will consist "
               "of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an "
               "HTTP request, please output an 'exception'. Only output 'abnormal' or 'normal', no additional output "
               "is required. The given HTTP request is as follows:\n",
        "EVD": "Given the following traffic data <packet> that contains protocol fields, traffic features, "
               "and payloads. Please conduct the encrypted VPN detection task to determine which behavior or "
               "application category the VPN encrypted traffic belongs to. The categories include 'aim, bittorrent, "
               "email, facebook, ftps, hangout, icq, netflix, sftp, skype, spotify, vimeo, voipbuster, youtube'.\n",
        "TBD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and "
               "payloads. Please conduct the TOR BEHAVIOR DETECTION TASK to determine which behavior or application "
               "category the traffic belongs to under the Tor network. The categories include 'audio, browsing, chat, "
               "file, mail, p2p, video, voip'.\n"
    }
    # MET-LLM: structure the traffic section into [CLS] header [HEAD] payload [BODY]
    embedder = TrafficEmbeddingProcessor()
    structured = traffic_data
    try:
        if "<packet>" in traffic_data:
            pkt = "<packet>".join(traffic_data.split("<packet>")[1:])
            header, payload = embedder.split(pkt)
            structured = embedder.to_structured(header, payload)
    except Exception:
        pass
    if task == "AAD":
        prompt = prepromt_set[task] + structured.split("<packet>:")[-1]
    else:
        prompt = prepromt_set[task] + structured
    return prompt
def chat_llm(model, tokenizer, text, max_new_tokens=256, top_p=0.8, temperature=0.8):
    engine = LLMInferenceEngine(InferenceConfig(model_path=""))
    engine.model = model
    engine.tokenizer = tokenizer
    engine._cfg.max_new_tokens = max_new_tokens
    engine._cfg.top_p = top_p
    engine._cfg.temperature = temperature
    return engine.chat(text)



def main(config, prompt: str = None, **kwargs):
    instruction_text, traffic_data = prompt_processing(prompt)

    with open(config, "r", encoding="utf-8") as fin:
        config = json.load(fin)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    try:
        model_config = AutoConfig.from_pretrained(config["model_path"], trust_remote_code=True)
    except Exception:
        model_config = None
    try:
        model = AutoModelForCausalLM.from_pretrained(config["model_path"], config=model_config, trust_remote_code=True)
    except Exception:
        model = AutoModel.from_pretrained(config["model_path"], config=model_config, trust_remote_code=True)

    # Stage 1: task understanding
    ptuning_path = os.path.join(config["peft_path"], config["peft_set"]["NLP"])
    model_nlp = load_model(model, ptuning_path)

    model_nlp = model_nlp.eval()

    response, history = chat_llm(model_nlp, tokenizer, instruction_text)
    print(response)

    # Stage 2: task-specific traffic learning
    task = config["tasks"][response]
    ptuning_path = os.path.join(config["peft_path"], config["peft_set"][task])
    model_downstream = load_model(model, ptuning_path)

    model_downstream = model_downstream.eval()

    traffic_prompt = preprompt(task, traffic_data)
    response, history = chat_llm(model_downstream, tokenizer, traffic_prompt)
    print(response)


if __name__ == "__main__":
    fire.Fire(main)
