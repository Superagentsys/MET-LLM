from __future__ import annotations

import json
import os
import streamlit as st

from metllm.core.embedding import TrafficEmbeddingProcessor
from metllm.core.inference_engine import LLMInferenceEngine, InferenceConfig


def build_premise(task: str) -> str:
    prepromt_set = {
        "MTD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and payloads. Please conduct the ENCRYPTED MALWARE DETECTION TASK to determine which application category the encrypted beign or malicious traffic belongs to. The categories include 'BitTorrent, FTP, Facetime, Gmail, MySQL, Outlook, SMB, Skype, Weibo, WorldOfWarcraft,Cridex, Geodo, Htbot, Miuref, Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus'.\n",
        "BND": "Given the following traffic data <packet> that contains protocol fields, traffic features, and payloads. Please conduct the BOTNET DETECTION TASK to determine which type of network the traffic belongs to. The categories include 'IRC, Neris, RBot, Virut, normal'.\n",
        "WAD": "Classify the given HTTP request into normal and abnormal categories. Each HTTP request will consist of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an HTTP request, please output an 'exception'. Only output 'abnormal' or 'normal', no additional output is required. The given HTTP request is as follows:\n",
        "AAD": "Classify the given HTTP request into normal and abnormal categories. Each HTTP request will consist of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an HTTP request, please output an 'exception'. Only output 'abnormal' or 'normal', no additional output is required. The given HTTP request is as follows:\n",
        "EVD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and payloads. Please conduct the encrypted VPN detection task to determine which behavior or application category the VPN encrypted traffic belongs to. The categories include 'aim, bittorrent, email, facebook, ftps, hangout, icq, netflix, sftp, skype, spotify, vimeo, voipbuster, youtube'.\n",
        "TBD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and payloads. Please conduct the TOR BEHAVIOR DETECTION TASK to determine which behavior or application category the traffic belongs to under the Tor network. The categories include 'audio, browsing, chat, file, mail, p2p, video, voip'.\n",
    }
    return prepromt_set[task]


def run_app():
    with open("config.json", "r", encoding="utf-8") as fin:
        cfg = json.load(fin)

    st.set_page_config(page_title="MET-LLM Demo", page_icon=":robot:", layout='wide')

    engine = LLMInferenceEngine(InferenceConfig(model_path=cfg["model_path"]))
    embedder = TrafficEmbeddingProcessor()

    st.title("MET-LLM malicious detection system")

    human_instruction = st.text_area(label="User Instruction", height=100)
    traffic_data = st.text_area(label="Traffic Data", height=200)

    max_new_tokens = st.sidebar.slider('max_new_tokens', 0, 4096, 256, step=1)
    top_p = st.sidebar.slider('top_p', 0.0, 1.0, 0.8, step=0.01)
    temperature = st.sidebar.slider('temperature', 0.0, 1.0, 0.8, step=0.01)

    if st.button("Submit", key="predict"):
        premise = human_instruction
        task = cfg["tasks"].get(human_instruction, None)
        if task is None and "<packet>" not in traffic_data:
            st.error("Please provide instruction or valid <packet> section.")
            return

        if task is None:
            # try to ask LLM for task
            task_resp, _ = engine.chat(human_instruction)
            task = cfg["tasks"].get(task_resp, "MTD")

        prefix = build_premise(task)

        structured = traffic_data
        if "<packet>" in traffic_data:
            pkt = "<packet>".join(traffic_data.split("<packet>")[1:])
            header, payload = embedder.split(pkt)
            structured = embedder.to_structured(header, payload)

        full_prompt = prefix + structured
        engine._cfg.max_new_tokens = max_new_tokens
        engine._cfg.top_p = top_p
        engine._cfg.temperature = temperature

        out, _ = engine.chat(full_prompt)
        st.success(out)


if __name__ == "__main__":
    run_app()

import streamlit as st
from metllm.config import load_json_config
from metllm.models.loader import load_tokenizer_and_model
from metllm.infer.pipeline import run_inference


st.set_page_config(page_title="MET-LLM Server", page_icon=":robot:", layout='wide')


def main():
    cfg = load_json_config("config.json")
    tokenizer, model = load_tokenizer_and_model(cfg["model_path"])  # type: ignore

    st.title("MET-LLM Web 服务（重构版）")

    max_length = st.sidebar.slider('max_length', 0, 32768, 8192, step=1)
    top_p = st.sidebar.slider('top_p', 0.0, 1.0, 0.8, step=0.01)
    temperature = st.sidebar.slider('temperature', 0.0, 1.0, 0.8, step=0.01)

    human_instruction = st.text_area(label="任务描述", height=100, placeholder="请输入要执行的流量分析任务描述")
    traffic_data = st.text_area(label="Traffic Data", height=200, placeholder="请粘贴 <packet> 开头的流量文本或从工具生成")

    if st.button("提交"):
        if not human_instruction or not traffic_data:
            st.warning("任务描述与流量数据均不能为空")
        else:
            task, pred = run_inference(human_instruction, traffic_data, tokenizer, model, cfg)
            st.success(f"下游任务: {task}\n预测结果: {pred}")


if __name__ == "__main__":
    main()


