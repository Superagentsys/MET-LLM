import sys
import subprocess


def main():
    # 修正为当前仓库的 PEFT 路径
    cmd = [sys.executable, "peft/ea-peft.py", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()


