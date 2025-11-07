import sys
import subprocess


def main():
    # 统一切换到新的 DeepSeek LoRA 训练脚本
    cmd = [sys.executable, "metllm_core/training/finetune_lora.py", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()


