"""Monte Carlo orderings on the Anthropic judge faithfulness — Appendix F.

Draws one **judge faithfulness** score per cell (Anthropic vendor, since the paper
runs both variants on that judge), averages per format, and ranks the three
formats (json / vision / tooluse). Two variants:

  * ``--variant 24`` — the "8 redrawn cells per format" variant. Only the
    variance-study cells contribute (4 features × 2 XAIs × 3 formats = 24 cells);
    nothing is fixed. Per-format means average over the 8 sampled cells.
  * ``--variant 18`` — the "full 18" variant. All 9 features × 2 XAIs = 18 cells
    per format; the 8 covered by the variance study are resampled (uniform
    over its up-to-3 gens), the remaining 10 are fixed at their G2a
    Anthropic-judge faithfulness. Per-format means average over 18 cells.

Recipe: ``random.seed(42)``; each iteration, per cell in scope pick one draw
with ``random.choice`` over the cell's **sorted** gen list, average per format,
rank. Ties are reported explicitly (any two means equal ⇒ ``ties``); the strict
``first_place`` / ``last_place`` shares count iterations where all three means
differ. The stable-sort shares (``first_place_stable``) reproduce the paper's
run-1 numbers, which are computed with ``sorted([json, vision, tooluse],
key=-mean)`` so ties go to the earlier index.

Paper check values (run 1, from Appendix F):
  ``--variant 24``  ties ≈ 0.34; modal-ordering share 0.449; mean spread 0.448;
                    first_place_stable shape {0.672, 0.234, 0.095}.
  ``--variant 18``  first_place_strict = {tooluse 0.589, vision 0.130, json 0.009};
                    ties 0.272; last_place_stable {json 0.776, …};
                    pairwise tooluse > vision 0.704, vision > tooluse 0.149;
                    per-format SD in 0.074–0.099.
Both variants reproduce the paper's Appendix F run-1 values exactly; after the
rerun they are expected to change.

Excludes the documented truncated draw ``json_xgb_weekday_gen1``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
EXCLUDED = {"json_xgb_weekday_gen1"}
GEN_STEM_RE = re.compile(
    r"^(?P<form>vision|tooluse|json)_(?P<xai>ebm|xgb)_"
    r"(?P<feature>[a-z]+)_gen(?P<gen>[012])$"
)
G2A_STEM_RE = re.compile(r"^(?P<form>vision|tooluse|json)_(?P<xai>ebm|xgb)_"
                         r"(?P<feature>[a-z]+)$")

MODALITIES = ["json", "vision", "tooluse"]  # stable-sort order (paper's convention)


def _load_variance_judge_scores(
    judge_dir: Path,
) -> dict[tuple[str, str, str], list[float]]:
    """{(form, xai, feature): [faithfulness per gen, sorted by gen]}, excluding
    the exempt draw. `random.choice` in the outer loop is stable because gens
    are appended in sorted order."""
    per_cell: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for p in sorted(judge_dir.glob("*.json")):
        stem = p.stem
        if stem in EXCLUDED:
            continue
        m = GEN_STEM_RE.match(stem)
        if not m:
            continue
        rec = json.loads(p.read_text())
        f = rec.get("faithfulness")
        if f is None:
            continue
        per_cell[(m["form"], m["xai"], m["feature"])].append((int(m["gen"]), float(f)))
    return {k: [v for _, v in sorted(pairs)] for k, pairs in per_cell.items()}


def _load_g2a_judge_scores(
    judge_dir: Path,
) -> dict[tuple[str, str, str], float]:
    """{(form, xai, feature): G2a Anthropic-judge faithfulness}."""
    out: dict[tuple[str, str, str], float] = {}
    for p in sorted(judge_dir.glob("*.json")):
        m = G2A_STEM_RE.match(p.stem)
        if not m:
            continue
        rec = json.loads(p.read_text())
        f = rec.get("faithfulness")
        if f is None:
            continue
        out[(m["form"], m["xai"], m["feature"])] = float(f)
    return out


def _all_differ(means: dict[str, float]) -> bool:
    return len(set(means.values())) == len(means)


def _rank_stable(means: dict[str, float]) -> tuple[str, ...]:
    """Stable sort in the paper's fixed modality order — ties go to the earlier
    index. Reproduces run-1's `first_place_stable` shares."""
    return tuple(sorted(MODALITIES, key=lambda m: -means[m]))


def run(variant: str, n_iter: int, seed: int = 42,
        judge: str = "anthropic") -> dict:
    if judge == "anthropic":
        var_dir = RESULTS / "global_variance_judge"
        g2a_dir = RESULTS / "global_judge"
    elif judge == "openai":
        var_dir = RESULTS / "global_variance_judge_openai"
        g2a_dir = RESULTS / "global_judge_openai"
    else:
        raise ValueError(judge)

    var_scores = _load_variance_judge_scores(var_dir)
    g2a_scores = _load_g2a_judge_scores(g2a_dir)

    if variant == "24":
        sample_cells = sorted(var_scores)                       # 24 cells
        fixed_cells: list[tuple[str, str, str]] = []
    elif variant == "18":
        sample_cells = sorted(var_scores)                       # 24 cells → 8 per format
        fixed_cells = sorted(k for k in g2a_scores if k not in var_scores)
    else:
        raise ValueError(variant)

    rng = random.Random(seed)
    orderings: Counter = Counter()
    first_stable: Counter = Counter()
    last_stable: Counter = Counter()
    first_strict: Counter = Counter()
    last_strict: Counter = Counter()
    ties = 0
    pairwise: Counter = Counter()
    per_iter_means: dict[str, list[float]] = defaultdict(list)
    per_iter_spreads: list[float] = []

    for _ in range(n_iter):
        picked: dict[tuple[str, str, str], float] = {}
        for key in sample_cells:
            picked[key] = rng.choice(var_scores[key])
        for key in fixed_cells:
            picked[key] = g2a_scores[key]

        means: dict[str, float] = {}
        for f in MODALITIES:
            vals = [v for k, v in picked.items() if k[0] == f]
            if vals:
                means[f] = sum(vals) / len(vals)
        if len(means) < len(MODALITIES):
            continue

        stable = _rank_stable(means)
        orderings[stable] += 1
        first_stable[stable[0]] += 1
        last_stable[stable[-1]] += 1
        if _all_differ(means):
            first_strict[stable[0]] += 1
            last_strict[stable[-1]] += 1
        else:
            ties += 1
        for a, b in itertools.combinations(MODALITIES, 2):
            if means[a] > means[b]:
                pairwise[(a, b)] += 1
            elif means[b] > means[a]:
                pairwise[(b, a)] += 1
        per_iter_spreads.append(max(means.values()) - min(means.values()))
        for m, v in means.items():
            per_iter_means[m].append(v)

    total = sum(orderings.values())

    def _pct(counter: Counter, denom: int) -> dict[str, float]:
        return {("_".join(k) if isinstance(k, tuple) else k):
                round(v / denom, 4) for k, v in counter.items()}

    modal_order, modal_count = orderings.most_common(1)[0]
    per_format_sd = {m: round(statistics.pstdev(vs), 4)
                     for m, vs in per_iter_means.items()}

    return {
        "variant": variant,
        "judge": judge,
        "n_iter": n_iter,
        "seed": seed,
        "n_cells_sampled": len(sample_cells),
        "n_cells_fixed": len(fixed_cells),
        "ties_share": round(ties / total, 4) if total else None,
        "first_place_stable": _pct(first_stable, total),
        "last_place_stable": _pct(last_stable, total),
        "first_place_strict": _pct(first_strict, total),
        "last_place_strict": _pct(last_strict, total),
        "pairwise_share": _pct(pairwise, total),
        "orderings_stable": _pct(orderings, total),
        "modal_ordering": "_".join(modal_order),
        "modal_share": round(modal_count / total, 4),
        "mean_spread": round(sum(per_iter_spreads) / len(per_iter_spreads), 4)
                       if per_iter_spreads else None,
        "per_format_sd": per_format_sd,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("24", "18"), default="18")
    ap.add_argument("--n", type=int, default=5000,
                    help="Iterations (paper: 5000).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge", choices=("anthropic", "openai"),
                    default="anthropic",
                    help="Which vendor's faithfulness to score on. Paper uses anthropic.")
    args = ap.parse_args()

    r = run(args.variant, args.n, seed=args.seed, judge=args.judge)
    text = json.dumps(r, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text)
        print(f"# {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
