from __future__ import annotations

import json
import os
import subprocess
import time
import streamlit as st

from metllm.core.embedding import TrafficEmbeddingProcessor
from metllm.core.inference_engine import LLMInferenceEngine, InferenceConfig


def tshark_dump_fields(tmp_pcap: str, display_filter: str, fields: list[str]) -> list[str]:
    extract_str = " -e " + " -e ".join(fields) + " "
    tmp_txt = "continuous_capture.txt"
    dump_cmd = f"tshark -r {tmp_pcap} {extract_str} -T fields -Y '{display_filter}' > {tmp_txt}"
    os.system(dump_cmd)
    if not os.path.exists(tmp_txt):
        return []
    with open(tmp_txt, "r") as fin:
        return fin.readlines()


def run_app():
    with open("configs/config.json", "r", encoding="utf-8") as fin:
        cfg = json.load(fin)

    st.set_page_config(page_title="MET-LLM Realtime", page_icon=":satellite:", layout='wide')

    engine = LLMInferenceEngine(InferenceConfig(model_path=cfg["model_path"]))
    embedder = TrafficEmbeddingProcessor()

    st.title("MET-LLM 实时流量检测")

    interfaces = subprocess.run(["tshark", "-D"], capture_output=True, text=True).stdout.strip().split('\n')
    selected_interface = st.selectbox("选择网络接口", interfaces)
    port_filter = st.number_input("过滤端口(可选)", min_value=1, max_value=65535, value=None, step=1)

    max_new_tokens = st.sidebar.slider('max_new_tokens', 0, 4096, 256, step=1)
    top_p = st.sidebar.slider('top_p', 0.0, 1.0, 0.8, step=0.01)
    temperature = st.sidebar.slider('temperature', 0.0, 1.0, 0.8, step=0.01)

    cols = st.columns([1, 2])
    with cols[0]:
        capture = st.button("开始捕获")
    with cols[1]:
        results = st.empty()

    if capture:
        tmp_pcap = "continuous_capture.pcap"
        filter_str = f"port {port_filter}" if port_filter else ""
        cap_cmd = f"tshark -i {selected_interface.split('.')[0]} -w {tmp_pcap}"
        if filter_str:
            cap_cmd += f" -f '{filter_str}'"
        proc = subprocess.Popen(cap_cmd, shell=True)

        try:
            last_size = 0
            fields = [
                "frame.encap_type", "frame.time", "frame.time_epoch", "frame.time_delta",
                "frame.number", "frame.len", "frame.protocols", "ip.version", "ip.len",
                "ip.ttl", "ip.proto", "tcp.srcport", "tcp.dstport", "tcp.payload",
                "udp.srcport", "udp.dstport", "data.len"
            ]
            display_filter = f"(tcp.port == {port_filter} or udp.port == {port_filter})" if port_filter else "tcp or udp"
            while True:
                time.sleep(1)
                if not os.path.exists(tmp_pcap):
                    continue
                size = os.path.getsize(tmp_pcap)
                if size == last_size:
                    continue
                last_size = size
                lines = tshark_dump_fields(tmp_pcap, display_filter, fields)
                if not lines:
                    continue
                pkt_line = lines[-1]
                values = pkt_line[:-1].split("\t")
                pkt = ", ".join([f"{f}: {v}" for f, v in zip(fields, values) if v])
                header, payload = embedder.split(pkt)
                structured = embedder.to_structured(header, payload)
                engine._cfg.max_new_tokens = max_new_tokens
                engine._cfg.top_p = top_p
                engine._cfg.temperature = temperature
                out, _ = engine.chat(structured)
                results.write(out)
        finally:
            proc.terminate()


if __name__ == "__main__":
    run_app()


