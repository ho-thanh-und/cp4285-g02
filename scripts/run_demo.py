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
    parser.add_argument("--home-pool", type=int, default=None,
                        help="SASRec products the home page's four discovery places are chosen from "
                             "(default: the exported, tested 20; 100 = the website setting, EXPERIMENT_LOG.md 6p)")
    parser.add_argument("--log", action="store_true",
                        help="Record anonymous impressions and clicks in logs/interactions.jsonl "
                             "(off by default; the page then shows a notice; HANDOFF.md section 4)")
    parser.add_argument("--click-order", choices=["first", "last"], default="first",
                        help="Order of a visit's re-clicked products for the home page: by first click "
                             "(as trained and tested) or by last click (the website setting, EXPERIMENT_LOG.md 6r)")
    args = parser.parse_args()
    run(args.handoff, args.port, args.home_pool, args.log, args.click_order)
