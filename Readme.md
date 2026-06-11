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
├── notebooks/       # 8 Jupyter notebooks (01–08)
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
| Poisson-log     | XGB   | 39.01 | 23.68 | 0.952 | 7.06         | 0          |
| Poisson-log     | EBM   | 48.37 | 27.00 | 0.926 | 9.72         | 0          |

**3 — Explanation generation** (`03_Explanations_Generation.ipynb`)
Global explanations (SHAP feature importance for XGB; term importances for EBM) and local explanations for 10 test instances stratified across `cnt` quintiles, stored as JSON plus waterfall-plot PNGs.

**4 — Three LLM pipelines**
All pipelines use `claude-sonnet-4-6` and produce three-part explanations (`[PREDICTION]`, `[DRIVERS]`, `[RECOMMENDATION]`) for non-technical staff.

- **`04` JSON → Text** — the LLM receives global importance and local SHAP/EBM contributions as structured JSON. System prompt cached via Anthropic prompt caching; raw values denormalized into plain language (e.g. `temp=0.68` → `~27.9 °C`) before the call.
- **`05` Vision → Text** — the LLM receives the instance's waterfall plot as a base64-encoded PNG and reads bar lengths visually (no numeric access to contribution values).
- **`06` Tool-Use** — the LLM retrieves data itself through 8 defined tools (feature schema, importance, prediction, SHAP values, partial dependence, value context, similar instances, counterfactuals) in an agentic loop — averaging **5.85 tool calls** per explanation.

**5 — Evaluation** (`07_Evaluation.ipynb`, `08_Evaluation_Ichmoukhamedov.ipynb`)
Quantitative cost/latency, keyword-based faithfulness, LLM-as-judge across three judge versions (uncalibrated Sonnet, calibrated-rubric Sonnet, independent Opus), and formal faithfulness metrics after Ichmoukhamedov et al. (Rank / Sign / Value Agreement).

| Pipeline   | Avg words | Input tok. | Output tok. | Cost (20 calls) | Avg latency |
| ---------- | --------- | ---------- | ----------- | --------------- | ----------- |
| JSON→Text  | 211       | 1,837      | 511         | $0.26           | 12.2 s      |
| Vision     | 207       | 2,187      | 512         | $0.28           | 12.3 s      |
| Tool-Use   | 306       | 3,427      | 1,263       | $0.58           | 29.8 s      |

## Key findings

1. **Completeness is robust** — all three pipelines score ≥ 4.9/5; the three-part structure is reliably maintained.
2. **Faithfulness depends on judge strictness** — Vision is consistent (≈ 4.4); JSON→Text and Tool-Use vary more across rubrics.
3. **JSON→Text is most efficient** (≈ $0.013 per explanation).
4. **Tool-Use produces longer, evidence-backed explanations** (+46% words, with partial-dependence and counterfactual support) at ~2.2× cost and ~2.4× latency.
5. **Vision** sits close to JSON→Text on cost/latency but has structurally lower faithfulness potential (bar lengths are read visually and imprecisely).
6. **Self-preference bias is measurable** — Opus scores sit systematically below Sonnet scores under an identical rubric, strongest for Tool-Use.

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