"""
utils/ – Gemeinsame Module für die Belegarbeit XAI.

Stellt Daten-, Modell- und Erklärungs-Loading-Logik zentral bereit,
sodass alle Notebooks (02-07) auf konsistenter Grundlage arbeiten.
"""

from pathlib import Path

# Wurzel-Verzeichnis des Projekts (Implementation/), unabhängig vom CWD.
# utils liegt unter Implementation/utils/, also ist parent.parent die Wurzel.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EXPLANATIONS_DIR = PROJECT_ROOT / "explanations"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Feste Test-Instanzen für lokale Erklärungen.
# Werden in allen drei LLM-Pipelines (04, 05, 06) verwendet,
# damit die Evaluation in 07 vergleichbar ist.
INSTANCE_IDS = [224, 580, 1041, 1481, 1677, 2058, 2510, 3543, 3847, 4454]

# Reproduzierbarkeit
RANDOM_STATE = 42

# Submodule-Exports (nach den Konstanten, um zirkuläre Importe zu vermeiden)
from .data import sample_stratified  # noqa: F401  (re-export for convenience)
from .judge import parse_judge_response, judge_batch_sc  # noqa: F401
from .generation import (  # noqa: F401
    run_resumable_generation,
    run_batch_generation,
    generation_filename,
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
    "sample_stratified",
    "parse_judge_response",
    "judge_batch_sc",
    "run_resumable_generation",
    "run_batch_generation",
    "generation_filename",
    "make_custom_id",
    "message_request",
    "submit_batch",
    "wait_for_batch",
    "collect_results",
    "run_batch",
]
