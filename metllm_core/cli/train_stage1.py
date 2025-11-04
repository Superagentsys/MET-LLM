import sys
import subprocess


def main():
    cmd = [sys.executable, "dual-stage-tuning/main.py", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()


