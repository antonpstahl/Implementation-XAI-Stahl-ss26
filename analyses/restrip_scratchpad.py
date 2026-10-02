"""Re-strip persisted `explanation` fields that leaked a `<thinking>` or
`<analysis>` scratchpad block.

Background: run 1's `strip_scratchpad` only matched `<analysis>`. Sonnet sometimes
emits `<thinking>` instead, so 14 G2a records + 13 variance draws + 4 G2b raw
records shipped the raw scratchpad into the persisted `explanation` and into the
judge input. The stripper is fixed at `utils/llm.py`; this script fixes the
persisted data:

  * records that are being *regenerated* anyway (XGB records that also need the
    new titled plots) are left alone, the C1 deletion cell handles them;
  * variance `gen0` records that are byte copies of a G2a record are deleted
    here so `04Gf` (`seed_generation_zero`) can re-seed them from the freshly
    re-stripped G2a record + its new judge verdicts;
  * every other leaked record is re-stripped in place, its rubric before/after
    is checked (must be equal, verified 2026-09-29), and its matching judge
    verdicts (both vendors) are deleted so the idempotent `run_global_judge`
    re-scores it;
  * the G2b raw `vision_all_ebm` and `vision_beeswarm_ebm` records are
    additionally re-split with `write_split_records`, and the script confirms
    that the split files are byte-identical to what was there before (per the
    plan, the leaked block sits before the first `[FEATURE:]` header and the
    splitter drops that preamble).

Dry-run by default (prints the plan and a rubric-parity table); `--apply` writes.
The rerun order in the plan is: this script → notebook 04Gc/04Gd/04Ge/05G/05Gb
regenerate the XGB records → 04Gf re-seeds the variance `gen0` files → 04Gf
draws + judges gen1/gen2 → the `test_no_scratchpad_leak.py` guard turns green.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.llm import strip_scratchpad
from utils.rubric import load_ground_truth, rubric_score
from utils.global_whole import (
    split_whole_model_record,
    write_split_records,
    WHOLE_CONDITIONS_BY_NAME,
)

RESULTS = ROOT / "results"
STAMP = "2026-09-29"


def _feature_names() -> list[str]:
    """The 9 predictors, in the training-column order, read from a global
    explanation JSON so the list stays in sync with the data pipeline."""
    p = ROOT / "explanations" / "global_ebm_poisson_log.json"
    return [f["feature"] for f in json.loads(p.read_text())["global_importance"]]

# Records that will be *regenerated* in Part C (they see the new titled plots),
# so B3 does not touch them, C1 deletes and C2/C4/C5 regenerate.
_REGEN_G2A = re.compile(r"^(vision|tooluse)_xgb_[a-z]+\.json$")
_REGEN_VAR = re.compile(r"^(vision|tooluse)_xgb_[a-z]+_gen[012]\.json$")
_REGEN_WHOLE_RAW = re.compile(r"^(vision_all|tooluse_all|vision_beeswarm)_xgb\.json$")

# Variance gen0 files are byte copies of the corresponding G2a record. If leaked,
# delete + let 04Gf re-seed from the freshly re-stripped G2a record so the byte
# identity of (G2a record ↔ variance gen0 record) and their verdicts is preserved.
_RESEED_VAR_GEN0 = re.compile(r"^(vision|tooluse|json)_ebm_[a-z]+_gen0\.json$")

# Truncated draw, documented exemption.
_EXEMPT = {"global_variance/json_xgb_weekday_gen1.json"}

_LEAK_OPEN = re.compile(r"<(analysis|thinking)>")  # case-sensitive: matches strip_scratchpad

JUDGE_DIRS = {
    "global": [RESULTS / "global_judge", RESULTS / "global_judge_openai"],
    "global_variance": [
        RESULTS / "global_variance_judge",
        RESULTS / "global_variance_judge_openai",
    ],
    "global_whole": [
        RESULTS / "global_whole_judge",
        RESULTS / "global_whole_judge_openai",
    ],
}


def _classify(rel_path: str) -> str:
    """Return one of: 'exempt', 'skip_regen', 'reseed_gen0', 'restrip',
    'restrip_whole_raw', 'unknown'."""
    if rel_path in _EXEMPT:
        return "exempt"
    top, _, name = rel_path.partition("/")
    if top == "global":
        return "skip_regen" if _REGEN_G2A.match(name) else "restrip"
    if top == "global_variance":
        if _REGEN_VAR.match(name):
            return "skip_regen"
        if _RESEED_VAR_GEN0.match(name):
            return "reseed_gen0"
        return "restrip"
    if top == "global_whole":
        if _REGEN_WHOLE_RAW.match(name):
            return "skip_regen"
        return "restrip_whole_raw"
    return "unknown"


def _iter_leaked():
    for top in ("global", "global_variance", "global_whole"):
        d = RESULTS / top
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            expl = rec.get("explanation")
            if not isinstance(expl, str) or not _LEAK_OPEN.search(expl):
                continue
            rel = p.relative_to(RESULTS).as_posix()
            yield p, rel, rec


def _rubric_parity(rec: dict, before: str, after: str) -> tuple[dict, dict] | None:
    """Score the record with both explanations; None for records the rubric skips
    (G2b raw records don't carry `feature`/`xai_model` in the same shape)."""
    if "xai_model" not in rec or "feature" not in rec:
        return None
    try:
        gt = load_ground_truth(rec["xai_model"], rec["feature"])
    except Exception:
        return None
    return rubric_score(before, gt), rubric_score(after, gt)


def _judge_paths_for(rel: str) -> list[Path]:
    top, _, name = rel.partition("/")
    return [d / name for d in JUDGE_DIRS.get(top, [])]


def _rewrite_raw_and_check_splits(
    raw_path: Path, rec: dict, stripped: str, *, apply: bool
) -> tuple[bool, list[str]]:
    """For G2b raw records: verify that re-splitting with the stripped explanation
    yields byte-identical split files (the leaked block sits before the first
    [FEATURE:] header, so the splitter drops it). Returns (identical, changed_rel)."""
    if rec.get("condition") not in WHOLE_CONDITIONS_BY_NAME:
        return True, []
    features = _feature_names()
    split_dir = RESULTS / "global_whole_split"
    before_splits = split_whole_model_record(rec, features=features)
    stripped_rec = dict(rec, explanation=stripped)
    after_splits = split_whole_model_record(stripped_rec, features=features)
    changed = []
    for b, a in zip(before_splits, after_splits):
        if b != a:
            changed.append(a["feature"])
    if apply and not changed:
        write_split_records([stripped_rec], features=features, split_dir=split_dir)
    return not changed, changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Write changes. Default is a dry run.")
    args = ap.parse_args()

    total = {"restrip": 0, "restrip_whole_raw": 0, "reseed_gen0": 0,
             "skip_regen": 0, "exempt": 0, "unknown": 0}
    parity_rows: list[str] = []
    parity_mismatch = 0
    split_mismatch: list[tuple[str, list[str]]] = []
    verdicts_deleted = 0
    gen0_deleted = 0

    for path, rel, rec in _iter_leaked():
        kind = _classify(rel)
        total[kind] += 1

        if kind in ("skip_regen", "exempt", "unknown"):
            continue

        before = rec["explanation"]
        after = strip_scratchpad(before)
        if after == before:
            # Unclosed tag: the stripper can't remove it (documented). Skip.
            print(f"! {rel}: leak is unclosed; skipping (would need regeneration)")
            continue
        removed = len(before) - len(after)

        parity = _rubric_parity(rec, before, after)
        if parity is not None:
            r_before, r_after = parity
            same = all(r_before[k] == r_after[k]
                       for k in ("direction", "rank", "structure", "total", "parsed_rank"))
            if not same:
                parity_mismatch += 1
            marker = "OK " if same else "!! "
            parity_rows.append(
                f"  {marker}{rel:60s}  -{removed:5d}c  "
                f"total {r_before['total']:.4f} -> {r_after['total']:.4f}  "
                f"rank {r_before['parsed_rank']} -> {r_after['parsed_rank']}"
            )
        else:
            parity_rows.append(f"  -- {rel:60s}  -{removed:5d}c  (raw whole-model record)")

        if kind == "reseed_gen0":
            # Delete + let 04Gf re-seed from the fresh G2a state.
            if args.apply:
                path.unlink()
                gen0_deleted += 1
                for jp in _judge_paths_for(rel):
                    if jp.exists():
                        jp.unlink()
                        verdicts_deleted += 1
            continue

        if kind == "restrip_whole_raw":
            identical, changed = _rewrite_raw_and_check_splits(
                path, rec, after, apply=args.apply
            )
            if not identical:
                split_mismatch.append((rel, changed))
            if args.apply:
                rec["explanation"] = after
                rec["scratchpad_restripped"] = STAMP
                path.write_text(
                    json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            continue

        # kind == "restrip"
        if args.apply:
            rec["explanation"] = after
            rec["scratchpad_restripped"] = STAMP
            path.write_text(
                json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            for jp in _judge_paths_for(rel):
                if jp.exists():
                    jp.unlink()
                    verdicts_deleted += 1

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"\n=== restrip_scratchpad ({mode}) ===")
    for k, v in total.items():
        print(f"  {k:20s} {v}")
    print("\nParity table (rubric before / after, must be equal):")
    for row in parity_rows:
        print(row)
    print(f"\nRubric mismatches: {parity_mismatch}")
    print(f"Judge verdicts deleted: {verdicts_deleted}")
    print(f"Variance gen0 records deleted (04Gf will re-seed): {gen0_deleted}")
    if split_mismatch:
        print("\n!! G2b split changes (plan says these must be byte-identical):")
        for rel, feats in split_mismatch:
            print(f"    {rel}: {feats}")
        return 1
    if parity_mismatch:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
