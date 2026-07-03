"""
utils/ - shared modules for the XAI thesis.

Provides data, model and explanation loading logic centrally so that all
notebooks (02 to 07) work on a consistent basis.
"""

from pathlib import Path

# Project root (Implementation/), independent of the CWD.
# utils lives under Implementation/utils/, so parent.parent is the root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EXPLANATIONS_DIR = PROJECT_ROOT / "explanations"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Fixed test instances for local explanations (validity sample, n = 20).
# 10 instances x 2 XAI models (xgb/ebm). Used in all pipelines (04La/04Lb/04Lc/04Ld)
# and carry the n=20 validity analysis in NB 05 (judge + inter judge agreement,
# judge sensitivity). Deliberately frozen.
INSTANCE_IDS = [224, 580, 1041, 1481, 1677, 2058, 2510, 3543, 3847, 4454]

# Reproducibility
RANDOM_STATE = 42

# Submodule exports (after the constants, to avoid circular imports)
from .judge import parse_judge_response, judge_batch_sc  # noqa: F401
from .generation import (  # noqa: F401
    run_resumable_generation,
    generation_filename,
    load_local_explanation,
    load_global_explanation,
    build_generation_record,
)
from .batch import (  # noqa: F401
    make_custom_id,
    message_request,
    submit_batch,
    wait_for_batch,
    collect_results,
    run_batch,
)
from .global_feature import (  # noqa: F401
    GLOBAL_RESULTS_SUBDIR,
    list_global_features,
    feature_importance_map,
    load_global_curve,
    shape_plot_path,
    beeswarm_plot_path,
    build_feature_json_payload,
    global_generation_filename,
    build_global_record,
    run_resumable_global_generation,
)
from .global_tools import (  # noqa: F401
    GLOBAL_TOOL_DEFINITIONS,
    GlobalToolBox,
    ToolImage,
    run_global_tool_use_loop,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "EXPLANATIONS_DIR",
    "RESULTS_DIR",
    "PROMPTS_DIR",
    "INSTANCE_IDS",
    "RANDOM_STATE",
    "parse_judge_response",
    "judge_batch_sc",
    "run_resumable_generation",
    "generation_filename",
    "load_local_explanation",
    "load_global_explanation",
    "build_generation_record",
    "make_custom_id",
    "message_request",
    "submit_batch",
    "wait_for_batch",
    "collect_results",
    "run_batch",
    "GLOBAL_RESULTS_SUBDIR",
    "list_global_features",
    "feature_importance_map",
    "load_global_curve",
    "shape_plot_path",
    "beeswarm_plot_path",
    "build_feature_json_payload",
    "global_generation_filename",
    "build_global_record",
    "run_resumable_global_generation",
    "GLOBAL_TOOL_DEFINITIONS",
    "GlobalToolBox",
    "ToolImage",
    "run_global_tool_use_loop",
]
