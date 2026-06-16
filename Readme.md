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
├── notebooks/       # 10 Jupyter notebooks (00 baseline, 01–08)
├── prompts/         # Prompt templates
└── utils/           # Python helper modules (data, models, explanations, llm, tools)
```

## Pipeline

**1 — Data preprocessing** (`01_Data_Preprocessing.ipynb`)
UCI Bike Sharing dataset (17,379 hourly observations, 2011–2012). Leakage and redundant features removed; multicollinearity handled (`atemp` vs. `temp`, r ≈ 0.99); categorical encoding for native splits; log1p target transform; 70/30 train/test split. Nine features remain (`hr`, `mnth`, `weekday`, `weathersit`, `yr`, `holiday`, `temp`, `hum`, `windspeed`).

**2 — Modeling** (`02a_Modeling_AllOptions.ipynb`, `02b_Comparison.ipynb`)
XGBoost and EBM (InterpretML), each trained with three loss functions. Poisson-log was selected for all downstream steps (best Poisson deviance, no negative predictions).

| Loss            | Model | RMSE  | MAE   | R²    | Poisson dev. | Neg. pred. |
| --------------- | ----- | ----- | ----- | ----- | ------------ | ---------- |
| Poisson-log     | XGB   | 45.44 | 27.00 | 0.935 | 9.38         | 0          |
| Poisson-log     | EBM   | 48.20 | 28.20 | 0.927 | 10.81        | 0          |

(Test set n = 5,227; values from `results/model_metrics_poisson_log.json`.)

**3 — Explanation generation** (`03_Explanations_Generation.ipynb`)
Global explanations (SHAP feature importance for XGB; term importances for EBM) and local explanations for 10 test instances stratified across `cnt` quintiles, stored as JSON plus waterfall-plot PNGs.

**4 — Three LLM pipelines + a deterministic baseline**
All LLM pipelines use `claude-sonnet-4-6` and produce three-part explanations (`[PREDICTION]`, `[DRIVERS]`, `[RECOMMENDATION]`) for non-technical staff.

- **`00` Template (baseline)** — a deterministic text-block generator that fills the same three-part structure from the identical SHAP/EBM JSON, with no LLM call. Answers the standard reviewer question: *what does the LLM add over a template?*
- **`04` JSON → Text** — the LLM receives global importance and local SHAP/EBM contributions as structured JSON. System prompt cached via Anthropic prompt caching; raw values denormalized into plain language (e.g. `temp=0.68` → `~27.9 °C`) before the call.
- **`05` Vision → Text** — the LLM receives the instance's waterfall plot as a base64-encoded PNG and reads bar lengths visually (no numeric access to contribution values).
- **`06` Tool-Use** — the LLM retrieves data itself through 8 defined tools (feature schema, importance, prediction, SHAP values, partial dependence, value context, similar instances, counterfactuals) in an agentic loop — averaging **5.65 tool calls** per explanation.

**5 — Evaluation** (`07_Evaluation.ipynb`, `08_Evaluation_Ichmoukhamedov.ipynb`)
Quantitative cost/latency, LLM-as-judge across three judge versions (uncalibrated Sonnet, calibrated-rubric Sonnet, independent Opus), and formal faithfulness metrics after Ichmoukhamedov et al. (Rank / Sign / Value Agreement).

Quantitative + LLM-judge (v1) summary across 20 explanations per pipeline (2 XAI models × 10 instances):

| Pipeline   | Avg words | Input tok.¹ | Output tok. | Cost (20 calls) | Avg latency | Judge Faith. | Clarity | Complete. |
| ---------- | --------- | ----------- | ----------- | --------------- | ----------- | ------------ | ------- | --------- |
| Template   | 54        | 0           | 0           | $0.00           | 0.0 s       | 5.00         | 4.70    | 4.00      |
| JSON→Text  | 208       | 616         | 510         | $0.16           | 11.7 s      | 4.35         | 4.90    | 4.95      |
| Vision     | 212       | 2,167       | 528         | $0.29           | 12.3 s      | 3.80         | 4.55    | 4.75      |
| Tool-Use   | 305       | 3,489       | 1,225       | $0.58           | 28.8 s      | 4.40         | 3.95    | 4.90      |

¹ *Input tokens are the billed, non-cached count. JSON→Text caches the system prompt (cache-read tokens, billed at ~10%, are not counted here), which is why its input count is far below Vision's freshly-sent image tokens.* Values from `results/eval_summary.csv`.

Formal faithfulness after Ichmoukhamedov et al. (NB 08, n = 10 instances; precision-style metrics — see limitation in NB 08 §4.1):

| Pipeline  | Rank Agr. | Sign Agr. | Value Agr. |
| --------- | --------- | --------- | ---------- |
| JSON→Text | 0.562     | 0.721     | 0.667      |
| Tool-Use  | 0.558     | 0.733     | 0.733      |
| Vision    | 0.429     | 0.679     | 0.575      |

> **Status of these findings:** descriptive/exploratory. With n = 10–20 explanations per pipeline, no repeated sampling and no inferential statistics yet, the differences below are **not** statistically confirmed (see the limitations table in `07_Evaluation.ipynb` §7). Treat them as directional.

1. **The deterministic template wins on faithfulness** — Template scores 5.00 vs. the LLM pipelines' 3.80–4.40. By construction it lists exactly the true top drivers; the LLMs trade some faithfulness for richer, more readable narratives. This is the central "what does the LLM add over a template?" result — and the trade-off, not a free lunch.
2. **Among LLM pipelines, faithfulness ranks Tool-Use (4.40) ≈ JSON→Text (4.35) > Vision (3.80)** — consistent with the formal Rank-Agreement (Vision 0.43 vs. ~0.56 for the others): reading bar lengths from a plot is structurally less precise than numeric access.
3. **Clarity and Completeness are at the ceiling for the LLM pipelines** (≥ 4.55 / ≥ 4.75) and do not discriminate between them; the template lags on completeness (4.00, thinner operational recommendation).
4. **JSON→Text is most efficient** (≈ $0.008 per explanation, lowest latency 11.7 s) — system-prompt caching keeps billed input tokens low.
5. **Tool-Use produces the longest, evidence-backed explanations** (+47% words vs. JSON→Text, with partial-dependence and counterfactual support) at ~3.6× cost and ~2.5× latency.
6. **Vision** matches JSON→Text on latency but costs more (image tokens, no caching benefit) and has the lowest faithfulness of all pipelines.
7. **Possible self-preference bias** — Opus (v3) scores sit systematically below Sonnet (v1/v2) under an identical rubric. This is *consistent with* a self-preference effect but not conclusive: both judges are Anthropic models, so a true cross-vendor judge is still outstanding (see implementation plan, Phase 2).

## Setup

```bash
# dependencies (Python 3.13.1)
pip install -r requirements.txt

# API key (do NOT commit your .env)
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

Run the notebooks in order (`01` → `08`). Paths are relative to the project root; reproducibility is fixed via `RANDOM_STATE = 42`.

## LLM configuration

All LLM calls use the **Anthropic Messages API** (accessed **2026-06-11**).
Parameters are centralised in `utils/llm.py`.

| Use case | Model | `max_tokens` | `temperature` |
|---|---|---|---|
| Explanation generation (NB 04 / 05 / 06) | `claude-sonnet-4-6` | 2048 | default (1.0) |
| Faithfulness check (NB 07) | `claude-sonnet-4-6` | 300 | default (1.0) |
| Judge v1 uncalibrated (NB 07) | `claude-sonnet-4-6` | 600 | default (1.0) |
| Judge v2 calibrated (NB 07) | `claude-sonnet-4-6` | 600 | default (1.0) |
| Judge v3 independent (NB 07) | `claude-opus-4-8` | 600 | default (1.0) |
| Ichmoukhamedov metrics (NB 08) | `claude-sonnet-4-6` | 700 | default (1.0) |

**Reproducibility note (→ Paper limitation):** Anthropic model IDs are versioned snapshots, but API behaviour (sampling, default parameters, tokenisation) can change silently between SDK releases. Results are tied to `anthropic==0.98.1` and the access date above. Future runs against the same model ID are not guaranteed to produce identical outputs.

## Notes

- The `.env` file and any API keys are excluded from version control and must be supplied locally.
- This repository contains the author's own code, data preparation, and results. Third-party publications are not redistributed here.

## Context

Term project (*Studienarbeit*), Information Systems, TU Dresden — supervised by Prof. Dr. Patrick Zschech. A follow-up Diplom thesis (master's-thesis equivalent) extends this work.