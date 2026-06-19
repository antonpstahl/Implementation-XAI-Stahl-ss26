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

# Feste Test-Instanzen für lokale Erklärungen (Validitäts-Sample, n = 20).
# 10 Instanzen × 2 XAI-Modelle (xgb/ebm). Werden in allen Pipelines (00/04/05/06)
# verwendet und tragen die n=20-Validitätsanalyse in NB 07 (v1/v2/v3/v4/v5,
# Inter-Judge-Agreement, Judge-Sensitivität). Bewusst **eingefroren** — der
# Skalierungslauf (Phase 3b) nutzt INSTANCE_IDS_SCALE, damit die teuren
# Validitäts-Judges nie versehentlich auf n≈200 mitlaufen.
INSTANCE_IDS = [224, 580, 1041, 1481, 1677, 2058, 2510, 3543, 3847, 4454]

# Reproduzierbarkeit
RANDOM_STATE = 42

# ── Phase 3b — Skalierung ────────────────────────────────────────────────────
# Stichprobengröße und Generationen pro Einheit für den Vollauf.
SCALE_N             = 5   # Testinstanzen (stratifiziert) für den 3b-Vollauf
N_GENERATIONS_SCALE = 3     # Generationen/Instanz für 04/05/06 (LLM-Stochastik)
                            # Template (00) ist deterministisch → dort 1 Generation.


def scale_instance_ids(n: int = SCALE_N, seed: int = RANDOM_STATE) -> list[int]:
    """Seeded, stratifizierte Test-Instanz-IDs für den Phase-3b-Vollauf.

    Zieht `n` Instanzen aus dem Test-Set, stratifiziert über cnt-Quintil,
    Tageszeit-Block (hr // 6) und Wetterlage (siehe `utils.data.sample_stratified`).
    Deterministisch bei festem `seed`. Lazy implementiert (lädt die Daten erst
    beim Aufruf), damit der Modulimport ohne Datendateien gelingt.

    Reproduzierbarkeit: Der Aufruf `scale_instance_ids()` ersetzt das fest
    verdrahtete `INSTANCE_IDS` für die Skalierung (Phase-3b-DoR) — statt 200
    Magic-Numbers im Code zu pinnen, wird die seeded Funktion bei jedem Lauf
    identisch ausgewertet.
    """
    from .data import load_train_test, sample_stratified
    _, _, X_test, y_test = load_train_test()
    return sample_stratified(X_test, y_test, n=n, seed=seed)

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
    "SCALE_N",
    "N_GENERATIONS_SCALE",
    "scale_instance_ids",
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
