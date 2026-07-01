# LLM-Assisted XAI Explanations for ML Predictions

Comparing how the **choice of XAI method** and the **handover format** to a large language model (LLM) affect the quality of automatically generated, natural-language explanations of ML predictions.

This repository accompanies a term project (*Studienarbeit*) at **TU Dresden**, supervised by **Prof. Dr. Patrick Zschech** (Chair of Business Information Systems, esp. Intelligent Systems and Services).

## Overview

The project studies how LLMs can translate predictions from machine-learning models into natural-language explanations for non-expert end users. The application case is the **Capital Bikeshare** system in Washington, D.C. — an hourly bike-rental demand dataset.

Two questions are examined in parallel:

1. **XAI method** — do explanations grounded in an *inherently interpretable* model (EBM shape functions) beat *post-hoc* explanations (SHAP on XGBoost)?
2. **Handover format** — does the LLM produce better explanations when it receives the information as **structured JSON**, as an **image** (waterfall plot, PNG), or through **active tool calls** (tool-use)?

## Repository structure

```
.
├── data/            # Raw data and prepared train/test splits
├── models/          # Trained models (6 .pkl files)
├── explanations/    # SHAP / EBM explanations as JSON + waterfall plots (PNG)
├── results/         # Pipeline outputs, evaluation plots, CSV summaries
├── notebooks/       # 12 Jupyter notebooks (01–08; incl. 02a/b and 04a–d)
├── prompts/         # Prompt templates
└── utils/           # Python helper modules (data, models, explanations, llm, tools)
```

## Pipeline

**1 — Data preprocessing** (`01_Data_Preprocessing.ipynb`)
UCI Bike Sharing dataset (17,379 hourly observations, 2011–2012). Leakage and redundant features removed; multicollinearity handled (`atemp` vs. `temp`, r ≈ 0.99); categorical encoding for native splits; log1p target transform; 70/30 train/test split. Nine features remain (`hr`, `mnth`, `weekday`, `weathersit`, `yr`, `holiday`, `temp`, `hum`, `windspeed`).

**2 — Modeling** (`02a_Modeling_AllOptions.ipynb`, `02b_Comparison.ipynb`)
XGBoost and EBM (InterpretML), each trained with three loss functions. Poisson-log was selected for all downstream steps (best Poisson deviance, no negative predictions).

<!-- AUTO-TABLE:model-metrics -->
| Loss        | Model | RMSE  | MAE   | R²    | Poisson dev. | Neg. pred. |
| ----------- | ----- | ----- | ----- | ----- | ------------ | ---------- |
| Poisson-log | XGB   | 45.44 | 27.00 | 0.935 | 9.38         | 0          |
| Poisson-log | EBM   | 48.20 | 28.20 | 0.927 | 10.81        | 0          |
<!-- /AUTO-TABLE:model-metrics -->

(Test set n = 5,227; values from `results/model_metrics_poisson_log.json`.)

**3 — Explanation generation** (`03_Explanations_Generation.ipynb`)
Global explanations (SHAP feature importance for XGB; term importances for EBM) and local explanations for 10 test instances stratified across `cnt` quintiles, stored as JSON plus waterfall-plot PNGs.

**4 — Three LLM pipelines + a deterministic baseline**
All LLM pipelines use `claude-sonnet-4-6` and produce three-part explanations (`[PREDICTION]`, `[DRIVERS]`, `[RECOMMENDATION]`) for non-technical staff.

- **`04a` Template (baseline)** — a deterministic text-block generator that fills the same three-part structure from the identical SHAP/EBM JSON, with no LLM call. Answers the standard reviewer question: *what does the LLM add over a template?*
- **`04b` JSON → Text** — the LLM receives global importance and local SHAP/EBM contributions as structured JSON. System prompt cached via Anthropic prompt caching; raw values denormalized into plain language (e.g. `temp=0.68` → `~27.9 °C`) before the call.
- **`04c` Vision → Text** — the LLM receives the instance's waterfall plot as a base64-encoded PNG and reads bar lengths visually (no numeric access to contribution values).
- **`04d` Tool-Use** — the LLM retrieves data itself through 8 defined tools (feature schema, importance, prediction, SHAP values, partial dependence, value context, similar instances, counterfactuals) in an agentic loop — averaging **5.65 tool calls** per explanation.

**5 — Evaluation** (`05_Evaluation.ipynb`, `06_Evaluation_Ichmoukhamedov.ipynb`)
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

Formal faithfulness after Ichmoukhamedov et al. (NB 06, n = 10 instances; precision-style metrics — see limitation in NB 06 §4.1):

<!-- AUTO-TABLE:faithfulness -->
| Pipeline     | Rank Agr. | Sign Agr. | Value Agr. |
| ------------ | --------- | --------- | ---------- |
| JSON to Text | 1.000     | 1.000     | 1.000      |
| Tool Use     | 0.988     | 1.000     | 1.000      |
| Vision       | 0.846     | 1.000     | 1.000      |
<!-- /AUTO-TABLE:faithfulness -->

**6 — Error analysis** (frozen diagnostic, kept locally — not included in the repository)
The 30 lowest faithfulness explanations were hand coded into an error taxonomy, separating genuine explanation errors (for example `yr` sign errors, near tie rank swaps) from extractor artefacts. The two dominant explanation error classes (yr sign and rank order) are fixed directly in the main generation prompts, so the main run already uses the corrected prompts. This was a **frozen diagnostic** that motivated those fixes; its learnings are now baked into the prompts (`pipeline_04/05/06`, `judge_system`) and guarded by regression tests (`tests/test_prompt_golden.py`, `tests/test_faithfulness.py`), so the notebook itself is no longer part of the tracked pipeline.

> **Status of these findings:** descriptive/exploratory. With n = 10–20 explanations per pipeline, no repeated sampling and no inferential statistics yet, the differences below are **not** statistically confirmed (see the limitations table in `05_Evaluation.ipynb` §7). Treat them as directional.

1. **The deterministic template wins on faithfulness** — Template scores 5.00 vs. the LLM pipelines' 3.80–4.40. By construction it lists exactly the true top drivers; the LLMs trade some faithfulness for richer, more readable narratives. This is the central "what does the LLM add over a template?" result — and the trade-off, not a free lunch.
2. **Among LLM pipelines, faithfulness ranks Tool-Use (4.40) ≈ JSON→Text (4.35) > Vision (3.80)** — consistent with the formal Rank-Agreement (Vision 0.43 vs. ~0.56 for the others): reading bar lengths from a plot is structurally less precise than numeric access.
3. **Clarity and Completeness are at the ceiling for the LLM pipelines** (≥ 4.55 / ≥ 4.75) and do not discriminate between them; the template lags on completeness (4.00, thinner operational recommendation).
4. **JSON→Text is most efficient** (≈ $0.008 per explanation, lowest latency 11.7 s) — system-prompt caching keeps billed input tokens low.
5. **Tool-Use produces the longest, evidence-backed explanations** (+47% words vs. JSON→Text, with partial-dependence and counterfactual support) at ~3.6× cost and ~2.5× latency.
6. **Vision** matches JSON→Text on latency but costs more (image tokens, no caching benefit) and has the lowest faithfulness of all pipelines.
7. **Judge robustness** — the judge runs on two independent models (Opus as primary, OpenAI gpt-4o-mini as a cross vendor check) under an identical rubric. Since neither judge is the generation model (Sonnet), self preference bias is avoided by design; agreement between the two judges indicates how stable the ranking is.

## Setup

```bash
# dependencies (Python 3.13.1)
pip install -r requirements.txt

# API key
echo "ANTHROPIC_API_KEY=sk..." > .env
```

Run the notebooks in order (`01` → `08`). Paths are relative to the project root; reproducibility is fixed via `RANDOM_STATE = 42`. The pipeline runs on the n = 20 validity sample (10 instances × 2 XAI models).

## LLM configuration

All LLM calls use the **Anthropic Messages API** (accessed **2026-06-11**).
Parameters are centralised in `utils/llm.py`.

| Use case | Model | `max_tokens` | `temperature` |
|---|---|---|---|
| Explanation generation (NB 04b / 04c / 04d) | `claude-sonnet-4-6` | 2048 | default (1.0) |
| Judge, primary (NB 05) | `claude-opus-4-8` | 900 | 0.0 |
| Judge, cross vendor (NB 05) | `gpt-4o-mini` (OpenAI) | 900 | default |
| Ichmoukhamedov metrics (NB 06) | `claude-sonnet-4-6` | 700 | default (1.0) |

**Reproducibility note (→ Paper limitation):** Anthropic model IDs are versioned snapshots, but API behaviour (sampling, default parameters, tokenisation) can change silently between SDK releases. Results are tied to `anthropic==0.98.1` and the access date above. Future runs against the same model ID are not guaranteed to produce identical outputs.

## Test suite

```bash
pytest tests/        # run all tests
pytest tests/test_prompt_golden.py -v   # prompt regression only
```

The suite covers sampling determinism, generation-loop persistence/resume, judge-parsing robustness, statistical functions, denormalization consistency, README consistency, and **prompt-fix regression**.
The prompt regression test (`test_prompt_golden.py`) freezes the SHA-256 hashes and key constraint phrases of all three pipeline prompts as corrected in Phase 3 (sign- and rank-fidelity rules for `yr=0`).
It is a hard gate: a fresh generation run must not start until all tests are green.

**Test status:** `pytest tests/` → **251 passed** (2026-06-29, Python 3.13).

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

Term project (*Studienarbeit*), Information Systems, TU Dresden — supervised by Prof. Dr. Patrick Zschech. A follow-up Diplom thesis (master's-thesis equivalent) extends this work.
