from typing import Tuple, Dict, Any
from metllm.traffic_utils import (
    load_metllm_config,
    split_header_payload,
    build_structured_sequence,
)
from metllm.models.loader import chat_llm


PREPROMPTS = {
    "MTD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and payloads. Please conduct the ENCRYPTED MALWARE DETECTION TASK to determine which application category the encrypted beign or malicious traffic belongs to. The categories include 'BitTorrent, FTP, Facetime, Gmail, MySQL, Outlook, SMB, Skype, Weibo, WorldOfWarcraft,Cridex, Geodo, Htbot, Miuref, Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus'.\n",
    "BND": "Given the following traffic data <packet> that contains protocol fields, traffic features, and payloads. Please conduct the BOTNET DETECTION TASK to determine which type of network the traffic belongs to. The categories include 'IRC, Neris, RBot, Virut, normal'.\n",
    "WAD": "Classify the given HTTP request into normal and abnormal categories. Each HTTP request will consist of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an HTTP request, please output an 'exception'. Only output 'abnormal' or 'normal', no additional output is required. The given HTTP request is as follows:\n",
    "AAD": "Classify the given HTTP request into normal and abnormal categories. Each HTTP request will consist of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an HTTP request, please output an 'exception'. Only output 'abnormal' or 'normal', no additional output is required. The given HTTP request is as follows:\n",
    "EVD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and payloads. Please conduct the encrypted VPN detection task to determine which behavior or application category the VPN encrypted traffic belongs to. The categories include 'aim, bittorrent, email, facebook, ftps, hangout, icq, netflix, sftp, skype, spotify, vimeo, voipbuster, youtube'.\n",
    "TBD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and payloads. Please conduct the TOR BEHAVIOR DETECTION TASK to determine which behavior or application category the traffic belongs to under the Tor network. The categories include 'audio, browsing, chat, file, mail, p2p, video, voip'.\n",
}


def build_preprompt(task: str, traffic_data: str) -> str:
    cfg = load_metllm_config()
    structured = traffic_data
    try:
        if "<packet>" in traffic_data:
            pkt = "<packet>".join(traffic_data.split("<packet>")[1:])
            header, payload = split_header_payload(pkt)
            structured = build_structured_sequence(header, payload, cfg)
    except Exception:
        pass
    return PREPROMPTS[task] + structured


def run_inference(human_instruction: str, traffic_data: str, tokenizer, model, config: Dict[str, Any]) -> Tuple[str, str]:
    # Stage 1: 任务理解
    task_response, _ = chat_llm(model, tokenizer, human_instruction)
    task = config["tasks"][task_response] if task_response in config.get("tasks", {}) else "MTD"
    # Stage 2: 任务特定流量学习
    traffic_prompt = build_preprompt(task, traffic_data)
    final_response, _ = chat_llm(model, tokenizer, traffic_prompt)
    return task_response, final_response


