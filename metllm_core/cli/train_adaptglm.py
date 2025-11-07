import sys
import subprocess


def main():
    # adaptglm 已替换为 DeepSeek LoRA 训练
    cmd = [sys.executable, "metllm_core/training/finetune_lora.py", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()


