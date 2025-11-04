import sys
import subprocess


def main():
    cmd = [sys.executable, "EA-PEFT/ea-peft.py", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()


