# LLM-Assisted XAI Explanations for ML Predictions

Comparing how the **choice of XAI method** and the **handover format** to a large language model (LLM) affect the quality of automatically generated, natural-language explanations of ML predictions.

This repository accompanies a term project (*Studienarbeit*) at **TU Dresden**, supervised by **Prof. Dr. Patrick Zschech** (Chair of Business Information Systems, esp. Intelligent Systems and Services).

## Overview

The project studies how LLMs can translate predictions from machine-learning models into natural-language explanations for non-expert end users. The application case is the **Capital Bikeshare** system in Washington, D.C., an hourly bike-rental demand dataset.

Two questions are examined in parallel:

1. **XAI method**: do explanations grounded in an *inherently interpretable* model (EBM shape functions) beat *post-hoc* explanations (SHAP on XGBoost)?
2. **Handover format**: does the LLM produce better explanations when it receives the information as **structured JSON**, as an **image** (plot, PNG), or through **active tool calls** (tool-use)?

### Two tracks

Following the 30 June supervision meeting the project runs on **two tracks**, and the
**global track is the main one**:

| Track | Unit of explanation | Notebooks | n | Role |
| --- | --- | --- | --- | --- |
| **Global** (main) | how the model uses a **feature**, and the **whole model** | `04Ga-04Ge`, `05G`, `05Gb` | 72 per-feature + 90 whole-model records | carries the findings |
| **Local** (comparison) | why **one instance** got its prediction | `04La-04Ld`, `05`, `06` | 20 explanations per pipeline | comparison basis, kept reproducible |

The global track scores every description against a **structured ground truth** (18
references, all 18 mechanically verified) with two independent scorers: a deterministic
keyword **rubric** and a reference-based **LLM judge**, the latter run across **two
vendors** (Anthropic Opus + OpenAI) as a robustness check.

## Repository structure

```
.
├── data/            # Raw data and prepared train/test splits
├── models/          # Trained models (6 .pkl files)
├── explanations/    # Local SHAP/EBM explanations + global curves, beeswarms, ground truth
├── results/         # Pipeline outputs, judge scores, evaluation plots, CSV summaries
├── notebooks/       # 18 Jupyter notebooks (01-06; 04Ga-Gf global, 04La-Ld local)
├── analyses/        # GT verification report, error analysis by shape type
├── planning/        # Revision plan, corrections log, limitations
├── prompts/         # Prompt templates (local + global + judge)
└── utils/           # Python helper modules (data, models, explanations, llm, tools,
                     #   global_feature, global_whole, groundtruth, rubric, global_eval)
```

`results/` holds both tracks side by side, so every number cited in the write-up is
reproducible from one tree:

```
results/
├── global/                    # G2a: 72 per-feature descriptions (4 forms x 2 models x 9 features)
├── global_rubric.csv          # G2a: deterministic rubric scores
├── global_judge{,_openai}/    # G2a: reference-based judge, 2 vendors
├── global_whole/              # G2b: 10 whole-model answers (5 conditions x 2 models)
├── global_whole_split/        # G2b: the same answers split into 90 per-feature records
├── global_whole_judge{,_openai}/   # G2b: judge on the split records, 2 vendors
├── pipeline00/04/05/06/       # local track: 20 explanations per pipeline
└── eval_*.{csv,json,png}, eval06_ichmoukhamedov/   # local-track evaluation
```

## Pipeline

**1. Data preprocessing** (`01_Data_Preprocessing.ipynb`)
UCI Bike Sharing dataset (17,379 hourly observations, 2011-2012). Leakage and redundant features removed; multicollinearity handled (`atemp` vs. `temp`, r ≈ 0.99); categorical encoding for native splits; log1p target transform; 70/30 train/test split. Nine features remain (`hr`, `mnth`, `weekday`, `weathersit`, `yr`, `holiday`, `temp`, `hum`, `windspeed`).

**2. Modeling** (`02a_Modeling_AllOptions.ipynb`, `02b_Comparison.ipynb`)
XGBoost and EBM (InterpretML), each trained with three loss functions. Poisson-log was selected for all downstream steps (best Poisson deviance, no negative predictions).

<!-- AUTO-TABLE:model-metrics -->
| Loss        | Model | RMSE  | MAE   | R²    | Poisson dev. | Neg. pred. |
| ----------- | ----- | ----- | ----- | ----- | ------------ | ---------- |
| Poisson-log | XGB   | 45.44 | 27.00 | 0.935 | 9.38         | 0          |
| Poisson-log | EBM   | 48.20 | 28.20 | 0.927 | 10.81        | 0          |
<!-- /AUTO-TABLE:model-metrics -->

(Test set n = 5,227; values from `results/model_metrics_poisson_log.json`.)

**3. Explanation generation** (`03_Explanations_Generation.ipynb`)
Global explanations (SHAP feature importance for XGB; term importances for EBM) and local explanations for 10 test instances stratified across `cnt` quintiles, stored as JSON plus waterfall-plot PNGs. The global track additionally derives, per feature, a **shape/dependence curve** and a **beeswarm** (`explanations/global_curve_*.json`, `explanations/plots/global/`); for the EBM, whose library has no beeswarm, one is **constructed** from the shape functions and validated against the XGB SHAP swarm (rank Spearman 0.883, direction agreement 100 %).

---

## Global track (main): how does the model use its features?

The unit of explanation is not a single prediction but the **model's use of a feature** — and, in `04Ge`, the **whole model at once**. Every description is scored against a structured ground truth.

**Ground truth** (`explanations/global_groundtruth/`, 18 references = 9 features × 2 XAI models). Per feature: shape type (`monotonic` / `non-monotonic` / `categorical` / `near-flat`), direction, importance rank, peak, top categories. Derived mechanically from the G0 curves via one shared `classify_shape` helper — the **same** helper the deterministic baseline uses, so the two cannot drift apart. All 18 references pass a mechanical re-derivation (`analyses/gt_verification.md`); a threshold sensitivity sweep leaves 15/18 labels stable across the whole grid, with the three border cases named explicitly.

**G2a — one feature per call** (`04Ga`–`04Gd`): four handover forms × 2 XAI models × 9 features = **72** descriptions, scored by the deterministic rubric *and* the reference-based judge (two vendors).

<!-- AUTO-TABLE:global-feature -->
| Form     | Rubric total | Judge Faith. | Clarity | Complete. | Faith. (OpenAI) |
| -------- | ------------ | ------------ | ------- | --------- | --------------- |
| Template | 0.944        | 4.50         | 4.06    | 5.00      | 4.72            |
| JSON     | 0.889        | 4.28         | 4.00    | 5.00      | 3.56            |
| Vision   | 0.898        | 4.39         | 3.50    | 5.00      | 3.94            |
| Tool Use | 0.898        | 4.39         | 3.44    | 5.00      | 4.06            |
<!-- /AUTO-TABLE:global-feature -->

The deterministic **template is not beaten** here: it leads nominally at 4.50, and **none** of the six pairwise comparisons is significant (Wilcoxon signed-rank on `(feature, xai_model)` pairs, Holm-corrected, all `p_adj = 1.0`, Cliff's *d* throughout "negligible"). Note the framing constraint: since the threshold unification the baseline hits the shape type *by construction*, which makes it a strict **reference floor** rather than an independent competitor. Rubric and judge correlate at Spearman 0.534 (n = 72, p < 0.001) — internal consistency between two ground-truth-bound scorers, not independent criterion validity.

The stable finding is not a modality ranking but a **failure mode**:

<!-- AUTO-TABLE:global-formtype -->
| Shape type    | n  | Judge Faith. | Clarity | Complete. |
| ------------- | -- | ------------ | ------- | --------- |
| near-flat     | 20 | 3.70         | 3.90    | 5.00      |
| categorical   | 28 | 4.43         | 3.68    | 5.00      |
| non-monotonic | 16 | 4.88         | 3.75    | 5.00      |
| monotonic     | 8  | 5.00         | 3.62    | 5.00      |
<!-- /AUTO-TABLE:global-formtype -->

**Over-attribution of negligible features** is the consistent error, converging across rubric, judge and the qualitative analysis (`analyses/error_examples_by_formtype.md`). The stratum sizes are too small for tests, so these are descriptive means.

**G2b — the whole model per call** (`04Ge`, exploratory): five conditions × 2 XAI models. Each answer must emit a rigid `[FEATURE: name] [EFFECT] … [IMPORTANCE] …` block per feature; `split_whole_model_record` cuts it into **90** per-feature records the same rubric and judge score unchanged. A feature the answer omits counts as a miss — which makes **coverage** directly measurable.

<!-- AUTO-TABLE:global-whole -->
| Condition       | Coverage (ebm · xgb) | Judge Faith. | Faith. (OpenAI) | Complete. |
| --------------- | -------------------- | ------------ | --------------- | --------- |
| json_all        | 9/9 · 9/9            | 4.72         | 3.39            | 3.61      |
| vision_all      | 9/9 · 9/9            | 4.22         | 3.06            | 2.94      |
| tooluse_all     | 9/9 · 9/9            | 4.89         | 3.72            | 3.89      |
| json_beeswarm   | 9/9 · 9/9            | 3.61         | 3.44            | 3.72      |
| vision_beeswarm | 9/9 · 9/9            | 3.89         | 2.72            | 4.00      |
<!-- /AUTO-TABLE:global-whole -->

Two axes this opens that the per-feature track cannot:

- **Mechanism (push vs pull)**, at constant information: pooled, `tooluse_all` (the model retrieves what it needs) beats the pushed conditions on the fair aggregate (0.935 / 0.926 vs. 0.671 / 0.769). **Disaggregated, that advantage is not uniform** — pooling mixes mechanism with modality:

| Push condition | XAI model | pull | push | pull − push |
| --- | --- | --- | --- | --- |
| `json_all` | ebm | 0.935 | 0.713 | +0.222 |
| `json_all` | xgb | 0.926 | 0.963 | **−0.037** |
| `vision_all` | ebm | 0.935 | 0.630 | +0.306 |
| `vision_all` | xgb | 0.926 | 0.574 | +0.352 |

  Pull beats the **image** push condition decisively on both models; against the **numeric** push condition it is a wash (`json_all` on XGB actually wins). The stable effect is therefore *"the image loses"*, not *"pull wins"* — report it that way, and treat "pull architecture" as a qualified recommendation rather than a headline. (`utils.global_eval.axis2_mechanism_pairwise`)
- **Representation**: `vision_all` collapses on **rank** (0.111 vs. 0.611 for `json_all` and 0.889 for `tooluse_all`) while holding up on direction (0.944). Reading nine separate plots conveys *shapes* but loses the *ordering* between features — the single beeswarm, which encodes rank as row order, scores 0.944–1.000 there.

> **Note on the coverage column.** The first `04Ge` run capped output at `MAX_TOKENS = 4096` and persisted no `stop_reason` for the JSON/vision paths, so `vision_all_xgb` and `vision_beeswarm_xgb` were silently truncated, showing 6/9 and 4/9. Both were regenerated at 16384 (`vision_all_xgb` needed 6503 output tokens — it was genuinely cut short) and now cover 9/9, as does every other condition. **Coverage therefore does not discriminate between these conditions at all**; the earlier gap was a token-limit artefact. `stop_reason` and `max_tokens` are now persisted on every path and `assert_not_truncated` refuses to store a truncated record.

> **A finding that did not survive.** The earlier "beeswarm image ≪ beeswarm numbers" result rested almost entirely on the truncated `vision_beeswarm_xgb` (fair_total 0.319). After the re-run it scores **0.917** — identical to the EBM variant, which was never truncated. The honest reading is a **small, consistent** modality gap (json_beeswarm 1.000 / 0.972 vs. vision_beeswarm 0.917 / 0.917), not the dramatic one first reported. The LLM reads the swarm image nearly as well as the equivalent numbers.

**Process metrics (G2a).** Straight off the persisted generation records:

<!-- AUTO-TABLE:global-process -->
| Form     | Avg input tok. | Avg output tok. | Avg latency | Avg tool calls |
| -------- | -------------- | --------------- | ----------- | -------------- |
| Template | 0              | 0               | 0.0 s       | n/a            |
| JSON     | 82,421         | 720             | 18.2 s      | n/a            |
| Vision   | 1,542          | 768             | 16.9 s      | n/a            |
| Tool Use | 69,274         | 884             | 22.1 s      | 3.1            |
<!-- /AUTO-TABLE:global-process -->

Two things to read here. First, **the modality comparison is not information-matched**: JSON receives the raw per-instance SHAP scatter (~12k points per feature) and Vision only the rendered plot — a **53:1** input-token ratio. The comparison therefore measures information *volume* as well as modality. G2b fixes this by aggregating the curve to a grid; G2a deliberately does not, and the asymmetry cuts in a useful direction: JSON had 53× the information and still did not beat the deterministic template. Second, **Tool-Use averages only 3.1 calls** — the model fetches the feature it was asked about and stops, rarely pulling rank context for comparison. That frugality is a finding in its own right, and it lines up with the rank collapse of `vision_all` on the whole-model task: rank information is what these pipelines most readily leave on the table.

**Generation variance (`04Gf`) — measured, and it changes how G2a must be read.** Every G2a number rests on one draw per cell at `temperature = 1.0`. `04Gf` re-drew 24 cells (`windspeed`, `holiday`, `weekday` — the near-flat stratum where all between-modality variation lives — plus `hr` as a ceiling control) three times each and scored every draw with both the rubric and both judges.

| Instrument | Within-cell sd | Between-modality range to resolve | Ratio |
| --- | --- | --- | --- |
| Rubric total | 0.089 | 0.055 | 1.6× |
| Judge faithfulness (Anthropic) | **0.662** | 0.222 | **3.0×** |
| Judge faithfulness (OpenAI) | 0.669 | 0.222 | 3.0× |

A single draw moves the score by three times the entire modality difference. Resampling one draw per cell 5,000 times, **all six possible modality orderings occur** (tool-use leads in 67 % of draws, JSON in 23 %, vision in 10 %); the mean best-to-worst spread per draw is 0.448, twice the 0.222 reported in `05G`.

Two consequences. The **null finding is strengthened** — "no detectable modality difference" holds for a harder reason than small effects: a single draw cannot resolve them at all. But any claim resting on the *nominal ordering* is an artefact of the one sample drawn and must not appear in the write-up. The ceiling control behaves as it should: `hr` shows sd = 0.000 under both the rubric and the Anthropic judge, so the variance sits entirely in the near-flat stratum. One caveat on the judge side: the OpenAI judge shows sd = 0.385 even on those control cells, where the rubric and the Anthropic judge both show zero — part of the measured spread is judge-side, so 0.662 is an **upper bound** on pure generation variance.

This is also the case a rubric-only study would have missed: 0.089 reads as reassuringly small, and only the judge reveals the scale.

**Judge robustness.** Both tracks are scored by two vendors under an identical rubric. Cross-vendor Krippendorff's α: per-feature Faithfulness 0.575 / Clarity 0.339 / Completeness 0.000 (constant 5 — a ceiling, not agreement); whole-model 0.329 / 0.584 / 0.623. The whole-model task still breaks the per-feature completeness ceiling, but the reliability is **moderate at best** — and notably lower than the 0.489 / 0.803 / 0.801 measured before the truncation fix. That drop is itself informative: the two truncated answers were scored low by *both* vendors, and that shared, artefact-driven agreement was inflating the reliability estimate. OpenAI scores Faithfulness systematically stricter (Δ ≈ 1.0 on the whole-model track) — a calibration offset, not a ranking disagreement: the ordering of conditions is the same under both vendors. Report **relative** comparisons, not absolute levels.

---

## Local track (comparison basis): why did *this* instance get *this* prediction?

**4. Three LLM pipelines + a deterministic baseline**
All LLM pipelines use `claude-sonnet-4-6` and produce three-part explanations (`[PREDICTION]`, `[DRIVERS]`, `[RECOMMENDATION]`) for non-technical staff.

- **`04La` Template (baseline)**: a deterministic text-block generator that fills the same three-part structure from the identical SHAP/EBM JSON, with no LLM call. It answers the standard reviewer question: *what does the LLM add over a template?*
- **`04Lb` JSON → Text**: the LLM receives global importance and local SHAP/EBM contributions as structured JSON. The system prompt is cached via Anthropic prompt caching; raw values are denormalized into plain language (e.g. `temp=0.68` → `~27.9 °C`) before the call.
- **`04Lc` Vision → Text**: the LLM receives the instance's waterfall plot as a base64-encoded PNG and reads bar lengths visually (no numeric access to contribution values).
- **`04Ld` Tool-Use**: the LLM retrieves data itself through 8 defined tools (feature schema, importance, prediction, SHAP values, partial dependence, value context, similar instances, counterfactuals) in an agentic loop, averaging **6.2 tool calls** per explanation (recomputed over `results/pipeline06/`, n = 20).

**5. Evaluation** (`05_Evaluation.ipynb`, `06_Evaluation_Ichmoukhamedov.ipynb`)
Quantitative cost/latency, LLM as judge with two independent judges (Opus as the primary judge plus OpenAI gpt-4o-mini as a cross vendor robustness check, identical rubric), and formal faithfulness metrics after Ichmoukhamedov et al. (Rank / Sign / Value Agreement).

Quantitative + LLM judge (Opus) summary across 20 explanations per pipeline (2 XAI models x 10 instances):

<!-- AUTO-TABLE:pipeline-eval -->
| Pipeline     | Avg words | Input tok.¹ | Output tok. | Cost (20 calls) | Avg latency | Judge Faith. | Clarity | Complete. |
| ------------ | --------- | ----------- | ----------- | --------------- | ----------- | ------------ | ------- | --------- |
| Template     | 57        | 0           | 0           | $0.00           | 0.0 s       | 5.00         | 4.15    | 4.80      |
| JSON to Text | 250       | 600         | 524         | $0.17           | 11.5 s      | 5.00         | 4.10    | 5.00      |
| Vision       | 239       | 1,085       | 571         | $0.19           | 13.2 s      | 4.50         | 4.00    | 5.00      |
| Tool Use     | 403       | 5,786       | 1,268       | $0.73           | 33.0 s      | 5.00         | 4.10    | 5.00      |
<!-- /AUTO-TABLE:pipeline-eval -->

¹ *Input tokens are the billed, non-cached count. JSON→Text caches the system prompt (cache-read tokens, billed at ~10%, are not counted here), which is why its input count is far below Vision's freshly-sent image tokens.* Values from `results/eval_summary.csv`.

Formal faithfulness after Ichmoukhamedov et al. (NB 06, n = 10 instances; precision-style metrics, see the limitation in NB 06 §4.1):

<!-- AUTO-TABLE:faithfulness -->
| Pipeline     | Rank Agr. | Sign Agr. | Value Agr. |
| ------------ | --------- | --------- | ---------- |
| JSON to Text | 1.000     | 1.000     | 1.000      |
| Tool Use     | 0.988     | 1.000     | 1.000      |
| Vision       | 0.846     | 1.000     | 1.000      |
<!-- /AUTO-TABLE:faithfulness -->

**6. Error analysis** (frozen diagnostic, kept locally, not included in the repository)
The 30 lowest faithfulness explanations were hand coded into an error taxonomy, separating genuine explanation errors (for example `yr` sign errors, near tie rank swaps) from extractor artefacts. The two dominant explanation error classes (yr sign and rank order) are fixed directly in the main generation prompts, so the main run already uses the corrected prompts. This was a **frozen diagnostic** that motivated those fixes; its learnings are now baked into the prompts (`pipeline_04/05/06`, `judge_system`) and guarded by regression tests (`tests/test_prompt_golden.py`), so the notebook itself is no longer part of the tracked pipeline.

> **Status of these findings:** descriptive/exploratory. With n = 20 explanations per pipeline, no repeated sampling and no inferential statistics, the differences below are **not** statistically confirmed (see the limitations table in `05_Evaluation.ipynb` §7). Treat them as directional. All numbers here come from the auto-generated tables above (`results/eval_summary.csv`); earlier README versions quoted a superseded judge run and are no longer accurate.

1. **Judge faithfulness barely separates the pipelines.** Template, JSON→Text and Tool-Use all sit at 5.00, Vision at 4.50 (the only pipeline with non-zero variance, sd 0.69). With three pipelines pinned to the scale maximum, faithfulness is at a **ceiling** on this task and cannot rank them.
2. **Vision is the one pipeline that loses information.** Its 4.50 matches the formal Rank-Agreement (0.846 vs. 1.000 for JSON→Text and 0.988 for Tool-Use): reading bar lengths off a waterfall plot is structurally less precise than numeric access. Sign and Value Agreement are 1.000 everywhere.
3. **Clarity does not discriminate either** (4.00–4.15 across all four); the template lags only on completeness (4.80 vs. 5.00 — a thinner operational recommendation).
4. **JSON→Text is the most efficient LLM pipeline** (≈ $0.009 per explanation, lowest latency 11.5 s), because system-prompt caching keeps billed input tokens low.
5. **Tool-Use produces the longest, evidence-backed explanations** (403 vs. 250 words, +61 % over JSON→Text, with partial-dependence and counterfactual support) at ~4.3× the cost and ~2.9× the latency, averaging 6.2 tool calls.
6. **Vision** costs slightly more than JSON→Text (image tokens, no caching benefit) at comparable latency, and is the weakest on faithfulness.
7. **This ceiling is why the global track exists.** A single-instance explanation is an easy task: it names three drivers, and every numeric pipeline gets them right. The differences the project is actually after only appear once the task is harder — describing a whole feature relationship, or the whole model at once. See the global track above.
8. **Judge robustness.** The judge runs on two independent models (Opus as primary, OpenAI gpt-4o-mini as a cross vendor check) under an identical rubric. Since neither judge is the generation model (Sonnet), self-preference bias is avoided by design.

## Setup

```bash
# dependencies (Python 3.13.1)
pip install -r requirements.txt

# API key
echo "ANTHROPIC_API_KEY=sk..." > .env
```

Paths are relative to the project root; reproducibility is fixed via `RANDOM_STATE = 42`.

```
01_Data_Preprocessing  →  02a/02b_Modeling  →  03_Explanations_Generation
                                                        │
        ┌───────────────────────────────────────────────┴───────────────┐
        │ global track (main)                                           │ local track (comparison)
        │ 04Ga template · 04Gb json · 04Gc vision · 04Gd tooluse        │ 04La · 04Lb · 04Lc · 04Ld
        │ 04Ge whole-model (5 conditions)                               │
        │ 05G  per-feature evaluation  ·  05Gb whole-model evaluation   │ 05 · 06 (Ichmoukhamedov)
        └───────────────────────────────────────────────────────────────┘
```

Every LLM notebook is guarded by a `RUN_API` flag (default `False`): the whole non-API path — prompt assembly, payload construction, splitter, coverage, rubric — is verified against a deterministic stub that writes nothing to `results/`. Set it to `True` only for the billed run. Generation is resumable: an existing output file is never regenerated, so an interrupted run resumes where it stopped.

## LLM configuration

All LLM calls use the **Anthropic Messages API** (accessed **2026-06-11**).
Parameters are centralised in `utils/llm.py`.

| Use case | Model | `max_tokens` | `temperature` |
|---|---|---|---|
| Global per-feature generation (NB 04Gb / 04Gc / 04Gd) | `claude-sonnet-4-6` | 2048 | default (1.0) |
| Global whole-model generation (NB 04Ge) | `claude-sonnet-4-6` | **16384** | default (1.0) |
| Global judge, primary (NB 05G / 05Gb) | `claude-opus-4-8` | 900 | 0.0 |
| Global judge, cross vendor (NB 05G / 05Gb) | `gpt-4o-mini` (OpenAI) | 900 | default |
| Local explanation generation (NB 04Lb / 04Lc / 04Ld) | `claude-sonnet-4-6` | 2048 | default (1.0) |
| Local judge, primary (NB 05) | `claude-opus-4-8` | 900 | 0.0 |
| Local judge, cross vendor (NB 05) | `gpt-4o-mini` (OpenAI) | 900 | default |
| Ichmoukhamedov metrics (NB 06) | `claude-sonnet-4-6` | 700 | default (1.0) |

The whole-model ceiling is deliberately 8× the per-feature one: one answer carries nine `[FEATURE]` blocks plus a recommendation. The first run used 4096 and truncated two answers **silently**, because `stop_reason` was not persisted on the JSON/vision paths — see the warning in the global-track section. Both fields are now written on every path and `assert_not_truncated` gates the write.

**Reproducibility note (→ Paper limitation):** Anthropic model IDs are versioned snapshots, but API behaviour (sampling, default parameters, tokenisation) can change silently between SDK releases. Results are tied to `anthropic==0.98.1` and the access date above. Future runs against the same model ID are not guaranteed to produce identical outputs.

## Test suite

```bash
pytest tests/        # run all tests
pytest tests/test_prompt_golden.py -v   # prompt regression only
```

The suite covers sampling determinism, generation-loop persistence/resume, judge-parsing robustness, statistical functions, denormalization consistency, README consistency, and **prompt-fix regression**. For the global track it additionally covers the constructed EBM beeswarm, the global curve artefacts, ground-truth derivation, the deterministic rubric, the whole-model payloads and the forced-schema splitter, and the **output-token truncation guards** (a record that hit the ceiling must never be persisted or scored).
The prompt regression test (`test_prompt_golden.py`) freezes the SHA-256 hashes and key constraint phrases of all three pipeline prompts as corrected in Phase 3 (sign- and rank-fidelity rules for `yr=0`).
It is a hard gate: a fresh generation run must not start until all tests are green.

The README consistency test is part of that gate: every numeric table in both READMEs is wrapped in `<!-- AUTO-TABLE:name -->` sentinels and regenerated from `results/` by `utils/update_readme_tables.py`. A number that drifts from its artefact fails the suite, which is what keeps two judge generations from being mixed in one document.

**Test status:** `pytest tests/` → **282 passed** (2026-09-07, Python 3.13).

**If a prompt is intentionally improved:**
1. Edit the prompt file.
2. Recompute the hash: `shasum -a 256 prompts/<file>.md`
3. Update `GOLDEN_HASHES` in `tests/test_prompt_golden.py`.
4. If the constraint phrase changed, update `REQUIRED_PHRASES` in the same file.
5. Confirm `pytest tests/test_prompt_golden.py` passes.

## Notes

- The `.env` file and any API keys are excluded from version control and must be supplied locally.
- This repository contains the author's own code, data preparation, and results. Third-party publications are not redistributed here.

## Context

Term project (*Studienarbeit*), Information Systems, TU Dresden, supervised by Prof. Dr. Patrick Zschech. A follow-up Diplom thesis (master's-thesis equivalent) extends this work.
