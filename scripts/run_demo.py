"""Run Clew from a source checkout; no training or export is performed."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session_rerank.demo import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clew local research demo")
    parser.add_argument("--handoff", type=Path, help="Final exported model folder")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(args.handoff, args.port)
