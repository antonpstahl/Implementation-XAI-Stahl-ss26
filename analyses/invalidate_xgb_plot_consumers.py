"""Delete every persisted record whose generation saw the old, un-titled XGB
plots, G2a XGB records for the vision/tooluse pipelines, the corresponding
variance draws, and the G2b whole-model conditions vision_all / tooluse_all /
vision_beeswarm on the XGB side. Judge verdicts are removed alongside so the
idempotent 05G/05Gb re-scores after regeneration.

Dry-run by default; ``--apply`` writes. The dry-run should report **210** files
(6 per-feature directories × 18 or 24 records = 126, plus 3 whole-model
conditions × 28 files = 84).

Run this in Part C1 of the fair-plots rerun plan, right before regenerating XGB
in 04Gc / 04Gd / 04Ge and re-seeding variance gen0 in 04Gf.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.global_whole import invalidate_whole_condition

R = ROOT / "results"

G2A = re.compile(r"^(vision|tooluse)_xgb_[a-z]+\.json$")
VAR = re.compile(r"^(vision|tooluse)_xgb_[a-z]+_gen[012]\.json$")

PER_FEATURE = {
    "global":                        G2A,
    "global_judge":                  G2A,
    "global_judge_openai":           G2A,
    "global_variance":               VAR,
    "global_variance_judge":         VAR,
    "global_variance_judge_openai":  VAR,
}

WHOLE = ["vision_all", "tooluse_all", "vision_beeswarm"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Delete files. Default is a dry run that only prints counts.")
    args = ap.parse_args()

    n = 0
    print(f"{'directory':40} {'files':>6}")
    print(f"{'-' * 40} {'-' * 6}")

    for sub, rx in PER_FEATURE.items():
        hits = [p for p in sorted((R / sub).glob("*.json")) if rx.match(p.name)]
        print(f"{sub:40} {len(hits):>6}")
        n += len(hits)
        if args.apply:
            for p in hits:
                p.unlink()

    for c in WHOLE:
        if args.apply:
            d = invalidate_whole_condition(c, "xgb", results_dir=R)
        else:
            # Mirror what invalidate_whole_condition would touch, without deleting.
            d = [R / "global_whole" / f"{c}_xgb.json"]
            for s in ("global_whole_split", "global_whole_judge",
                      "global_whole_judge_openai"):
                d += list((R / s).glob(f"{c}_xgb_*.json"))
            d = [p for p in d if p.exists()]
        print(f"G2b {c:36} {len(d):>6}")
        n += len(d)

    verb = "DELETED" if args.apply else "WOULD DELETE"
    print(f"\n{verb} {n} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
