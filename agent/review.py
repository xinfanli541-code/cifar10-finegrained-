"""错判日志汇总的独立入口，等价于 `python -m agent.cli review`。

用法：python -m agent.review [--log agent/logs/misclassified.jsonl]
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.cli import run_review  # noqa: E402
import argparse  # noqa: E402


def main():
    ap = argparse.ArgumentParser(prog="agent.review")
    ap.add_argument("--log", default=None)
    return run_review(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
