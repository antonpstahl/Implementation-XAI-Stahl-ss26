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
# 10 instances x 2 XAI models (xgb/ebm). Used in all pipelines (04a/04b/04c/04d)
# and carry the n=20 validity analysis in NB 05 (judge + inter judge agreement,
# judge sensitivity). Deliberately frozen: a larger scaling run uses
# scale_instance_ids so the expensive validity judges never run on it by accident.
INSTANCE_IDS = [224, 580, 1041, 1481, 1677, 2058, 2510, 3543, 3847, 4454]

# Reproducibility
RANDOM_STATE = 42

# --- Scaling -----------------------------------------------------------------
# Sample size and generations per unit for a larger run.
SCALE_N             = 5   # stratified test instances for the larger run
N_GENERATIONS_SCALE = 3     # generations per instance for 04/05/06 (LLM stochasticity)
                            # Template (00) is deterministic, so 1 generation there.


def scale_instance_ids(n: int = SCALE_N, seed: int = RANDOM_STATE) -> list[int]:
    """Seeded, stratified test instance IDs for a larger run.

    Draws `n` instances from the test set, stratified over the cnt quintile,
    the time of day block (hr // 6) and the weather situation (see
    `utils.data.sample_stratified`). Deterministic for a fixed `seed`. Lazy
    (loads the data only on call) so the module import works without data files.

    Reproducibility: `scale_instance_ids()` replaces the hard wired `INSTANCE_IDS`
    for scaling. Instead of pinning many magic numbers in the code, the seeded
    function is evaluated identically on every run.
    """
    from .data import load_train_test, sample_stratified
    _, _, X_test, y_test = load_train_test()
    return sample_stratified(X_test, y_test, n=n, seed=seed)

# Submodule exports (after the constants, to avoid circular imports)
from .data import sample_stratified  # noqa: F401  (re-export for convenience)
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

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "EXPLANATIONS_DIR",
    "RESULTS_DIR",
    "PROMPTS_DIR",
    "INSTANCE_IDS",
    "RANDOM_STATE",
    "SCALE_N",
    "N_GENERATIONS_SCALE",
    "scale_instance_ids",
    "sample_stratified",
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
]
