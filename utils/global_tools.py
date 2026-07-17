"""utils/global_tools.py - Tool-Use infrastructure for the global per-feature pipeline (04Gd).

Phase G2a Tool-Use: the LLM is asked about **one feature** (same unit as JSON/Vision) but
gets access to **all** tools - it must *pull* the relevant information itself (push vs.
pull). The tools expose the pre-computed G0/G1 artifacts, so no model is needed at
description time:

  * ``get_target_overview``   - the target + the full feature list (rank/importance).
  * ``get_feature_importances`` - the ranked importances (numeric beeswarm equivalent).
  * ``get_feature_curve``     - one feature's shape/dependence curve values (JSON equivalent).
  * ``get_feature_plot``      - one feature's shape/dependence PNG (Vision equivalent, image).
  * ``get_beeswarm_plot``     - the model's beeswarm PNG (image).

Design note (see planning/Revision_30062026_Umsetzungsplan.md, G2a): tool access is *not*
restricted to the asked feature - the "brave" (minimal) retrieval behaviour is a measured
outcome, not an enforced state. ``call_log`` is the process data (does it ground itself?
does it pull rank context / the beeswarm? how many calls?).

Mirrors :class:`utils.tools.ToolBox` (``dispatch`` -> ``_tool_{name}``, ``call_log``); the
loop mirrors the inline loop of the local 04Ld notebook but is parametrised over tools +
toolbox + model, and handles image tool results (a ``ToolImage`` result becomes an image
block in the ``tool_result`` content).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .global_feature import (
    _global_json,
    aggregate_curve,
    beeswarm_plot_path,
    feature_importance_map,
    load_global_curve,
    shape_plot_path,
)
from .llm import _encode_image, _with_retry


@dataclass
class ToolImage:
    """Marker for a tool result that is an image (not JSON), carrying its PNG path.

    ``label`` is what gets logged / put in the accompanying text block; the loop turns
    the path into a base64 image block for the ``tool_result``.
    """

    path: Path
    label: str


# Anthropic tool schemas. Kept deliberately small: two data tools + one importance/ranking
# tool + two image tools (per-feature plot and beeswarm), each individually retrievable.
GLOBAL_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_target_overview",
        "description": (
            "Return the prediction target (what the model predicts) and the full list "
            "of features with their global importance rank. Use this first to orient."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_feature_importances",
        "description": (
            "Return all features ranked by global importance (feature, importance, rank). "
            "The numeric equivalent of the beeswarm ordering - use it to judge how "
            "important a feature is relative to the others."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_feature_curve",
        "description": (
            "Return one feature's global shape/dependence curve as values: kind "
            "(continuous|categorical), x (feature values), y (contribution to the target "
            "in log space), plus its importance and rank."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"feature": {"type": "string", "description": "Feature name."}},
            "required": ["feature"],
        },
    },
    {
        "name": "get_feature_plot",
        "description": (
            "Return one feature's global shape/dependence plot as an image (the same PNG "
            "the vision pipeline sees). Read the curve shape, direction and peak visually."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"feature": {"type": "string", "description": "Feature name."}},
            "required": ["feature"],
        },
    },
    {
        "name": "get_beeswarm_plot",
        "description": (
            "Return the model's global beeswarm plot as an image: all features by "
            "importance, colour = feature value. Use it to place a feature in the overall "
            "ranking and read its direction."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def _preview(result: Any) -> str:
    if isinstance(result, ToolImage):
        return f"<image: {result.path.name}>"
    s = json.dumps(result, ensure_ascii=False)
    return s if len(s) <= 300 else s[:297] + "..."


class GlobalToolBox:
    """Serves the pre-computed global artifacts as tools for the 04Gd pipeline.

    Unlike :class:`utils.tools.ToolBox` (which runs the live model), this box only reads
    the G0/G1 files, so it needs no model object - just the artifact directories.
    """

    def __init__(
        self,
        model_name: str,
        *,
        explanations_dir: Path | str,
        plots_dir: Path | str,
        loss_key: str = "poisson_log",
        aggregate_curves: bool = False,
    ):
        self.model_name = model_name.lower()
        self.explanations_dir = Path(explanations_dir)
        self.plots_dir = Path(plots_dir)
        self.loss_key = loss_key
        # aggregate_curves=True collapses the raw XGB SHAP scatter (~12k points/feature)
        # to a grid before returning get_feature_curve. Off for the G2a per-feature
        # pipeline (one curve pulled, raw is fine); ON for the whole-model pipeline
        # (04Ge), where pulling several raw XGB curves would blow the context window.
        self.aggregate_curves = aggregate_curves
        self.call_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        """Run a tool call, log it, return the result (dict or :class:`ToolImage`)."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            result: Any = {"error": f"Unknown tool: '{name}'"}
        else:
            try:
                result = handler(**arguments)
            except Exception as exc:  # noqa: BLE001 - surfaced to the model as a result
                result = {"error": f"{type(exc).__name__}: {exc}"}

        self.call_log.append(
            {"tool": name, "arguments": arguments, "result_preview": _preview(result)}
        )
        return result

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------
    def _tool_get_target_overview(self) -> dict:
        g = _global_json(
            self.model_name, explanations_dir=self.explanations_dir, loss_key=self.loss_key
        )
        return {
            "target": g["task"],
            "model": self.model_name,
            "features": [
                {"feature": it["feature"], "rank": it["rank"]}
                for it in g["global_importance"]
            ],
        }

    def _tool_get_feature_importances(self) -> list[dict]:
        g = _global_json(
            self.model_name, explanations_dir=self.explanations_dir, loss_key=self.loss_key
        )
        return g["global_importance"]

    def _tool_get_feature_curve(self, feature: str) -> dict:
        curve = load_global_curve(
            self.model_name, feature, explanations_dir=self.explanations_dir
        )
        imp = feature_importance_map(
            self.model_name, explanations_dir=self.explanations_dir, loss_key=self.loss_key
        ).get(feature, {})
        if self.aggregate_curves:
            xs, ys = aggregate_curve(curve["x"], curve["y"])
            xs, ys = xs, [round(v, 5) for v in ys]
            value_space = "mean contribution per feature value, log space"
        else:
            xs, ys, value_space = curve["x"], curve["y"], "contribution to the target in log space"
        return {
            "feature": feature,
            "kind": curve["kind"],
            "x": xs,
            "y": ys,
            "value_space": value_space,
            "importance": imp.get("importance"),
            "rank": imp.get("rank"),
        }

    def _tool_get_feature_plot(self, feature: str) -> ToolImage:
        path = shape_plot_path(self.model_name, feature, plots_dir=self.plots_dir)
        if not path.exists():
            raise FileNotFoundError(f"No plot for feature '{feature}': {path}")
        return ToolImage(path=path, label=f"Global plot for feature '{feature}' ({self.model_name}).")

    def _tool_get_beeswarm_plot(self) -> ToolImage:
        path = beeswarm_plot_path(self.model_name, plots_dir=self.plots_dir)
        if not path.exists():
            raise FileNotFoundError(f"No beeswarm plot: {path}")
        return ToolImage(path=path, label=f"Global beeswarm plot ({self.model_name}).")


# -----------------------------------------------------------------------------
# Tool-Use loop (reusable, image-aware)
# -----------------------------------------------------------------------------

def _tool_result_content(result: Any) -> Any:
    """Build the ``tool_result`` content for a dispatch result.

    JSON results -> a JSON string; :class:`ToolImage` results -> a text label + a
    base64 image block (Anthropic accepts image content blocks in a tool_result).
    """
    if isinstance(result, ToolImage):
        return [
            {"type": "text", "text": result.label},
            _encode_image(result.path),
        ]
    return json.dumps(result, ensure_ascii=False)


def run_global_tool_use_loop(
    client: Any,
    toolbox: GlobalToolBox,
    *,
    user_message: str,
    system: str,
    model: str,
    max_tokens: int,
    tools: list[dict] = GLOBAL_TOOL_DEFINITIONS,
    max_rounds: int = 10,
) -> tuple[str, list[dict], int, int, str]:
    """Run the global Tool-Use loop for one feature.

    Reusable analogue of the inline loop in the local 04Ld notebook, parametrised over
    ``tools`` / ``toolbox`` / ``model`` and image-aware (plot tools return images).

    Returns
    -------
    (final_text, call_log, input_tokens, output_tokens, stop_reason)
        stop_reason: 'end_turn' | 'max_tokens' | 'max_rounds' | other API reason.
    """
    messages: list[dict] = [{"role": "user", "content": user_message}]
    total_in, total_out = 0, 0
    last_text = ""
    final_stop_reason = "max_rounds"

    for round_num in range(max_rounds):
        response = _with_retry(
            client.messages.create,
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
        )
        usage = response.usage
        total_in += usage.input_tokens
        total_out += usage.output_tokens

        messages.append({"role": "assistant", "content": response.content})

        candidate = next((b.text for b in response.content if hasattr(b, "text") and b.text), "")
        if candidate:
            last_text = candidate

        if response.stop_reason == "end_turn":
            return last_text, toolbox.call_log, total_in, total_out, "end_turn"

        if response.stop_reason != "tool_use":
            final_stop_reason = response.stop_reason
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = toolbox.dispatch(block.name, block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _tool_result_content(result),
                }
            )
            print(f"    [{round_num + 1}] {block.name}({list(block.input.keys())}) -> ok")

        messages.append({"role": "user", "content": tool_results})

    if final_stop_reason == "max_rounds":
        print(f"  [WARN] max_rounds={max_rounds} reached, using the last partial text.")
    else:
        print(f"  [WARN] stop_reason='{final_stop_reason}', using the last partial text.")
    return last_text or "[No answer]", toolbox.call_log, total_in, total_out, final_stop_reason
