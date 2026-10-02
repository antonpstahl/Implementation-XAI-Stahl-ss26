# Run-2 decision rules

> Written **before** the run-2 numbers are known, and committed under `analyses/`
> so the timestamp on record proves it. `planning/` is gitignored, so a rule
> written there is not evidence. If any threshold below is renegotiated after
> looking at the run-2 ledger, that change goes into git history with the
> reason, not on top of an already-known result.

Scope: fair-plots + scratchpad rerun (2026-09-29). Reference:
`planning/Fair_Plots_Rerun_Plan.md`, §D4 "Interpretation rules".

## RQ1: did the missing XGB plot title cause the whole-model rank collapse?

The single measurement is the G2b `vision_all` XGB "rank stated" count out of 9
(features whose rank the XGB whole-model answer names).

- **≥ 7/9 stated on XGB, with EBM still ≈ 9/9** → the rank-channel imbalance is
  gone. RQ1 becomes: "given an importance cue in each plot, the nine-plot
  handover states ranks on both arms." The run-1 → run-2 jump is reported as
  supporting (single-draw) evidence that the missing title caused the XGB
  collapse.
- **≤ 3/9 stated on XGB** → the title was **not** the cause. The paper's
  current channel explanation must go. Candidate reading: the XGB vertical
  range misleads (ρ = 0.23) and/or the scatter format. Rewrite RQ1 part 2 and
  Implication 1 accordingly.
- **4/9 – 6/9 stated on XGB (partial)** → report as partial. Run the optional
  redraws (D3, `results/global_whole_redraws/`, 4 extra `vision_all` draws per
  arm with the new plots) before writing the claim.

## Stated ranks: recount, do not carry over

The run-1 headline "82/82 stated ranks are exactly correct" must be **recounted**
from the run-2 G2b split records. Do not carry it over. Also check: did any
newly stated XGB rank become **wrong**? A single wrong rank flips the "always
correct when stated" claim.

Concretely: read `g2b.stated_total` and `g2b.stated_and_correct` from the
run-2 ledger (`analyses/numbers_ledger.py`); the two values must match, and both
must be recomputed, not copied from `analyses/ledger_run1/ledger.json`.

## Judge before/after

Run-1 judge scores for the XGB vision records were computed on leaked
scratchpad text. A judge-only before/after mixes two changes (plots + text).
Report judge numbers on the XGB arm only with that caveat, or not at all; the
rubric before/after is the clean comparison.

## Single-draw caveat

Every G2b cell is still **one draw per run**. The before/after on `vision_all`
is 1 vs 1 draw. The paper must say so and must not call it causal.
