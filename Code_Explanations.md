# Repository-Erklärung: LLM-gestützte XAI für Fahrradverleih-Prognosen

# 1. Repo-Überblick

**Zweck (sicher, aus [Readme.md](Readme.md)):** Forschungs-Implementierung einer Belegarbeit, die untersucht, wie Large Language Models (LLMs, hier Anthropic Claude Sonnet 4.6) XAI-Erklärungen (SHAP-Werte aus XGBoost; Term-Beiträge aus EBM) eines stündlichen Bikesharing-Regressionsmodells in natürlichsprachliche, dreigliedrige Erklärungen (`[VORHERSAGE]` / `[TREIBER]` / `[EMPFEHLUNG]`) für Nicht-Experten übersetzen.

**Verglichen werden drei LLM-Pipelines:**

| Pipeline | Eingabeform | Realisiert in |
|---|---|---|
| JSON → Text | strukturiertes JSON-Payload | [04_LLM_JSON_Pipeline.ipynb](notebooks/04_LLM_JSON_Pipeline.ipynb) |
| Vision → Text | Waterfall-Plot als base64-PNG | [05_LLM_Vision_Pipeline.ipynb](notebooks/05_LLM_Vision_Pipeline.ipynb) |
| Tool-Use (agentic loop) | LLM ruft eigene Tools | [06_LLM_ToolUse_Pipeline.ipynb](notebooks/06_LLM_ToolUse_Pipeline.ipynb) |

**Tech-Stack (sicher):** Python ≥ 3.10, pandas, numpy, xgboost, interpret (EBM via InterpretML), shap, joblib, anthropic SDK, python-dotenv, matplotlib/seaborn, optional sentence-transformers + transformers für Notebook 08.

**Einstiegspunkte:** Es gibt kein `main.py`. Der Workflow ist **notebook-getrieben** und linear (01 → 08).

**Wahrscheinlicher Datenfluss:**
```
data/hour.csv
   └─► 01_Preprocessing ─► data/train.csv, data/test.csv
        └─► 02a_Modeling ─► models/{xgb,ebm}_{loss}.pkl + results/model_metrics_{loss}.json
             └─► 02b_Comparison ─► results/model_comparison_summary.csv
                  └─► 03_Explanations ─► explanations/{global,local}_*.json
                       ├─► 04_JSON_Pipeline  ─► results/pipeline04/*.json
                       ├─► 05_Vision_Pipeline ─► explanations/plots/*.png + results/pipeline05/*.json
                       └─► 06_ToolUse_Pipeline ─► results/pipeline06/*.json
                            └─► 07_Evaluation, 08_Evaluation_Ichmoukhamedov ─► results/eval_*.{csv,png,json}
```

---

# 2. Architekturkarte

**Schichten (vom Kern nach außen):**

| Schicht | Inhalt | Verantwortung |
|---|---|---|
| Konfig | [utils/__init__.py](utils/__init__.py) | Pfadkonstanten, `INSTANCE_IDS`, `RANDOM_STATE` |
| Daten | [utils/data.py](utils/data.py) | Spalten-Schema, dtype-Wiederherstellung, `load_train_test()` |
| Modelle | [utils/models.py](utils/models.py) | Loss-Optionen, Metriken, Save/Load |
| Erklärungen | [utils/explanations.py](utils/explanations.py) | `FEATURE_SCHEMA`, `build_global/local`, SHAP-Cache |
| LLM-IO | [utils/llm.py](utils/llm.py) | Anthropic-Client, Retry, `ask_text`, `ask_with_images` |
| Tools | [utils/tools.py](utils/tools.py) | `TOOL_DEFINITIONS`, `ToolBox.dispatch` |
| Orchestrierung | `notebooks/*.ipynb` | Pipeline-Schritte + LLM-Aufrufe |

**Import-Topologie (sicher):**
```
__init__.py  ──┐
               ├── data.py        (DATA_DIR)
               ├── models.py      (MODELS_DIR)
               ├── explanations.py(EXPLANATIONS_DIR)
               ├── llm.py         (lädt .env)
               └── tools.py       (data.CATEGORICAL_COLS + explanations.FEATURE_SCHEMA)
```

Versteckte, **bewusste** Abhängigkeit: `tools.py` reicht `FEATURE_SCHEMA` aus `explanations.py` an Pipeline 06 weiter → Pipelines 04 und 06 sehen exakt dasselbe Schema.

**Wo liegt was?**
- Entry-Points → keine; Notebooks sind die „Mains"
- Konfiguration → [utils/__init__.py](utils/__init__.py) + `.env` (`ANTHROPIC_API_KEY`)
- Domain-Logik → [utils/explanations.py](utils/explanations.py) + [utils/tools.py](utils/tools.py)
- I/O → [utils/data.py](utils/data.py) (CSV), [utils/models.py](utils/models.py) (joblib), [utils/llm.py](utils/llm.py) (Anthropic SDK)
- Tests → **keine** (sicher beobachtet — keine `tests/`, kein `pytest`-Setup)

---

# 3. Erklärreihenfolge

1. `utils/__init__.py` → Konstanten
2. `utils/data.py` → Daten-Vertrag
3. `utils/models.py` → Loss-Registry + Metriken
4. `utils/explanations.py` → SHAP/EBM-Builder + Schema
5. `utils/llm.py` → Anthropic-Wrapper
6. `utils/tools.py` → Tool-Use-Backbone
7. Notebooks 01 → 08 in chronologischer Reihenfolge

---

# 4. Datei-Erklärungen

## Datei: [utils/__init__.py](utils/__init__.py)

**Zweck:** Single-Source-of-Truth für Pfadkonstanten und Reproduzierbarkeits-Konstanten.

### Block: Zeilen 8 + 12 — Projekt-Wurzel berechnen
```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```
- **[Originalfunktion]** Errechnet absolutes Projekt-Wurzelverzeichnis CWD-unabhängig.
- **[Syntax]** `__file__` ist Python-Built-in (Pfad zur aktuellen Datei); `.resolve()` macht ihn absolut und folgt Symlinks; `.parent.parent` geht zwei Ebenen hoch (`utils/__init__.py` → `utils/` → Wurzel).
- **[Semantik]** Alle abgeleiteten Pfade sind danach von jedem Aufrufort aus identisch.
- **[Risiken]** Bei Ausführung ohne `__file__` (z. B. `exec()`) schlägt es fehl — in dieser Forschungs-Pipeline unwahrscheinlich.

### Block: Zeilen 14–18 — Standard-Verzeichnisse
```python
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EXPLANATIONS_DIR = PROJECT_ROOT / "explanations"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
```
- **[Risiken]** `PROMPTS_DIR` zeigt auf einen Ordner, der **nicht existiert** (sicher beobachtet) — Altlast oder geplanter Ordner.

### Block: Zeilen 22–26 — Reproduzierbarkeitskonstanten
```python
INSTANCE_IDS = [224, 580, 1041, 1481, 1677, 2058, 2510, 3543, 3847, 4454]
RANDOM_STATE = 42
```
- **[Semantik]** **iloc-Positionsindizes** im `X_test`-Frame, stratifiziert über 5 cnt-Quintile gewählt ([Readme.md:108-122](Readme.md:108)). Alle drei Pipelines (04/05/06) erklären exakt diese 10 Instanzen → faire Vergleichbarkeit.
- **[Risiken]** Wenn Notebook 01 mit anderem Seed neu sampelt, zeigen die IDs auf andere Zeilen → stille Inkonsistenz.

### TL;DR
Pfade + 10 Test-Instanzen + Seed. Importiert von allen anderen `utils/*.py`-Modulen und jedem Notebook.

---

## Datei: [utils/data.py](utils/data.py)

**Zweck:** Einzige Stelle, an der nach dem CSV-Roundtrip die `category`-Dtypes wiederhergestellt werden — Kontraktbruch hier bricht XGBoost+EBM unten.

### Block: Zeilen 24–36 — Spalten-Klassifikation
```python
NOMINAL_COLS = ["weathersit"]
ORDINAL_COLS = ["mnth","hr","weekday"]
CATEGORICAL_COLS = NOMINAL_COLS + ORDINAL_COLS
```
- **[Semantik]** Trennung nominal vs. ordinal ist für Baummodelle redundant (beide nutzen native Categorical-Splits), aber sauber für künftige Lineare-Modelle und Plot-Logik.

### Block: Zeilen 38–44 — Numerische Spalten
- `yr` und `holiday` (binär 0/1) werden bewusst als `float64` geführt — Begründung im Kommentar: für Baummodelle äquivalent zu category, kein Informationsverlust.

### Block: Zeilen 49–62 — `DROP_COLS` (Safety-Net)
Wirft sicherheitshalber **immer** raus: Leakage (`casual`, `registered`), redundant (`season`, `workingday`), Multikollinearität (`atemp` r ≈ 0,99), IDs (`instant`, `dteday`), abgeleitetes Target (`cnt_log1p`).

### Block: Zeilen 70–87 — `_apply_dtypes(df)`
```python
df[col] = df[col].astype(float).astype(int).astype("category")
```
- **[Originalfunktion]** Drei-Stufen-Cast `float → int → category`.
- **[Warum hier?]** Direkt-Cast `astype("category")` würde bei pandas-`Int64`-nullable-Dtype Kategorien wie `1.0`, `2.0` erzeugen — bricht XGBoost 3.x mit `enable_categorical=True`. Der int-Zwischenschritt erzwingt saubere ganzzahlige Kategorien.
- **[Risiken]** `.astype(int)` crasht bei NaN — laut Daten ([Readme.md:36](Readme.md:36)) keine NaNs vorhanden, also OK.

### Block: Zeilen 101–142 — `load_train_test()`
- Lädt beide CSVs, ruft `_drop_unused` und `_apply_dtypes` auf, projiziert auf `FEATURE_COLS`, splittet in `(X_train, y_train, X_test, y_test)`.
- Drei explizite Fehler-Pfade mit klaren Hinweisen auf Notebook 01.

### TL;DR der Datei
**Der dtype-Kontrakt** des Repos. Jeder Bestandteil, der `cat.codes`, `cat.categories` oder `enable_categorical=True` verwendet, hängt von hier ab.

---

## Datei: [utils/models.py](utils/models.py)

**Zweck:** Loss-Optionen als immutable Registry, Metrik-Suite, Modell-Persistenz.

### Block: Zeilen 28–73 — `LossOption` + `LOSS_OPTIONS`
- **[Syntax]** `@dataclass(frozen=True)` → auto-generated, read-only Records.
- Drei Optionen: `squared_error` (RMSE/`reg:squarederror`, native contributions), `poisson_log` (Poisson-Dev/`count:poisson`, log contributions), `poisson_native` (gleiches Modell wie 2, aber approximative cnt-Skala-Beiträge).
- Das Feld **`contribution_space`** entscheidet später, wie LLM-Prompts die Beitragsdimension formulieren (Log-Raum → „multiplikativ via exp" vs. Originalskala → „additiv in Fahrrädern").

### Block: Zeilen 80–107 — `compute_metrics(y_true, y_pred)`
```python
rmse = sqrt(mean((y-ŷ)²))
mae  = mean(|y-ŷ|)
r2   = 1 - ss_res/ss_tot
poisson_deviance = 2 * mean(y*log((y+eps)/clip(ŷ,eps)) - (y - clip(ŷ,eps)))
```
- **[Kniff]** `np.clip(y_pred, eps, None)` gegen `log(0)`. Bei Squared-Error-Modellen mit negativen Predictions würde sonst Poisson-Deviance explodieren — genau das demaskiert die Untauglichkeit von Option 1.
- **[Zusätzlich]** `min_prediction` und `n_negative_predictions` → diagnostische Werte für Plausibilität.

### Block: Zeilen 114–121 — `save_model(model, model_type, loss_key)`
- Erzeugt `models/{xgb,ebm}_{poisson_log,poisson_native,squared_error}.pkl`.

### Block: Zeilen 131–173 — Legacy `save_ebm/load_ebm/save_xgb/load_xgb`
- **[Risiken]** Diese Funktionen suchen feste Dateinamen `ebm.pkl`/`xgb.pkl`, die **im aktuellen Repo nicht existieren** (sicher beobachtet). Toter Code / Refactor-Rückstand — sollte beseitigt werden.

### TL;DR
Saubere Loss-Registry + Metrik-Suite. Legacy-Pfad muss beim nächsten Refactor weg.

---

## Datei: [utils/explanations.py](utils/explanations.py)

**Zweck:** Brücke zwischen Modell-Internals und LLM-Eingabe-Format.

### Block: Zeilen 23–79 — `FEATURE_SCHEMA`
- Vollständiges deutschsprachiges Feature-Schema mit `weathersit`-Mapping (1=klar … 4=Gewitter), Range-Angaben und Denormalisierungs-Hinweisen für `temp`/`hum`/`windspeed`.
- Wird **roh** durchgereicht an Pipelines 04 (im JSON-Payload) und 06 (via Tool `get_feature_schema`).

### Block: Zeilen 96–104 — SHAP-Explainer-Cache
```python
_shap_cache: dict[int, Any] = {}
def _get_shap_explainer(model):
    model_id = id(model)
    if model_id not in _shap_cache:
        import shap
        _shap_cache[model_id] = shap.TreeExplainer(model)
    return _shap_cache[model_id]
```
- **[Risiken]** Cache-Key via `id(model)` ist unsicher unter GC (recyclte Adressen). Sicherer wäre `weakref.WeakKeyDictionary`.

### Block: Zeilen 116–123 — `_global_xgb`
- `np.abs(shap_vals).mean(axis=0)` ist die kanonische SHAP-Aggregation für globale Importance.

### Block: Zeilen 126–138 — `_global_ebm`
- Filtert Interaktionsterme (`" & " in name`) → globale Importance enthält **nur Haupteffekte**, alignt EBM mit additivem SHAP-Modell.

### Block: Zeilen 169–213 — `build_global(...)`
- Erzeugt JSON mit `model`, `task` (`TARGET_DESCRIPTION`), `feature_schema`, `metrics`, `base_value`, `global_importance` (gerankt + sortiert).
- **EBM-Sonderfall (Zeile 196)**: `base_value = float(np.log(model.predict(X_train)).mean())` — Näherung des Intercepts via Mittelwert der Log-Vorhersagen; technisch korrekt für Plotting, aber konzeptuell ist das echte Intercept-Attribut sauberer (wird aber für lokale Erklärungen via `d["extra"]["scores"][0]` gezogen).

### Block: Zeilen 216–279 — `build_local(...)`
- Erzeugt JSON mit `feature_values`, `y_true`, `prediction`, `base_value`, `contributions` (sortiert nach `|c|`).
- `contribution_space` hartkodiert auf `"log"` (Zeile 277). **Risiko:** bei künftigen Squared-Error-Pipelines falsch — sollte aus `LossOption.contribution_space` abgeleitet werden.
- Filter `if f in feature_values` (Zeile 265) wirft EBM-Interaktionsterme raus.

### TL;DR
Stellt das einheitliche JSON-Format her, das alle Folge-Notebooks lesen.

---

## Datei: [utils/llm.py](utils/llm.py)

**Zweck:** Anthropic-SDK-Wrapper mit Retry und Prompt-Caching.

### Block: Zeilen 22–27 — `.env` laden
- Defensiv: `ImportError` (kein `python-dotenv` installiert) wird verschluckt → fällt auf Shell-Env zurück.
- `override=True`: `.env` schlägt eine bereits gesetzte `ANTHROPIC_API_KEY` aus der Shell.

### Block: Zeilen 29–32 — Konstanten
```python
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 2048
_RETRYABLE = ("RateLimit", "APIStatusError", "APIConnectionError", "InternalServer", "Overloaded")
```

### Block: Zeilen 35–50 — `_with_retry`
- Exponentielles Backoff (Start 5 s, doppelt), max 2 Retries.
- **[Subtilität]** Substring-Match gegen `type(exc).__name__`. False-Positive-Risiko bei künftigen SDK-Klassen mit „RateLimit"-Substring. Besser: `isinstance` gegen `anthropic.APIError`-Hierarchie.
- **[Toter Code]** Zeile 50 (`raise last_exc`) wird nie erreicht.

### Block: Zeilen 76–111 — `ask_text(prompt, system=, cache_system=True)`
- **[Schlüssel-Mechanik]** Wenn `cache_system=True`, wird der System-Prompt als `[{"type":"text","text":system,"cache_control":{"type":"ephemeral"}}]` übergeben — Anthropic Prompt Caching.
- **[Voraussetzung]** Cache-Block braucht ≥ 1024 Tokens. Pipeline 04 hängt deswegen das vollständige Feature-Schema in den System-Prompt — überschreitet zuverlässig.

### Block: Zeilen 117–138 — `_encode_image(path)`
- Base64-Kodierung mit Media-Type-Map (`png/jpg/jpeg/gif/webp`).
- **[Risiken]** Keine Größenprüfung — versehentlich riesige PNGs würden API-Fehler oder hohe Kosten produzieren.

### Block: Zeilen 141–177 — `ask_with_images(prompt, image_paths, ...)`
- Content-Liste: **Bilder zuerst, Text zuletzt** (Anthropic-Best-Practice).
- Optional gecachter System-Prompt wie bei `ask_text`.

### TL;DR
Schlanker Wrapper. Tool-Use-Loop bewusst **nicht hier** — bleibt im Notebook 06, da projektspezifisch.

---

## Datei: [utils/tools.py](utils/tools.py)

**Zweck:** Tool-Definitionen + `ToolBox`-Klasse für Pipeline 06.

### Block: Zeilen 22–182 — `TOOL_DEFINITIONS`
Acht Tools im Anthropic-Format. Jede `description` erklärt Zweck (nicht nur Funktionalität) → bessere Tool-Auswahl durch das LLM:

| Tool | Zweck |
|---|---|
| `get_feature_schema` | Metadaten aller Features |
| `get_feature_importance` | globale Importance (top_k optional) |
| `get_prediction` | Was-wäre-wenn-Vorhersage |
| `get_shap_values` | lokale Beiträge einer Test-Instanz |
| `get_partial_dependence` | PD-Kurve für 1 Feature |
| `get_feature_value_context` | Percentile/Statistiken eines Werts |
| `get_similar_instances` | KNN-Nachbarn |
| `get_counterfactual_prediction` | Was-wäre-wenn bei Feature-Änderungen |

- **[Bug-Verdacht]** Die Beschreibung von `get_prediction` ([tools.py:60-65](utils/tools.py:60)) listet `season`/`workingday`/`yr` als kategorische Features. `season`/`workingday` existieren in den Daten **nicht mehr** — Dokumentations-Drift, das LLM könnte ungültige Features setzen.

### Block: Zeilen 188–229 — `ToolBox.__init__` + `dispatch`
- **[Pattern]** Reflection-Dispatching via `getattr(self, f"_tool_{name}")`.
- Logging in `self.call_log` (für Evaluation in 07/08).
- Alle Exceptions in `{"error": "..."}` umgewandelt → LLM bekommt strukturierten Fehler-Result, kein Crash.

### Block: Zeilen 234–263 — `_tool_get_feature_schema` + `_tool_get_feature_importance`
- XGB: SHAP mean(|·|) über X_train.
- EBM: `term_importances()` ohne Interaktionen.
- Optional `top_k`-Cap.

### Block: Zeilen 265–272 — `_tool_get_prediction`
- Baut DataFrame via `_build_input_df` (Modus-Fallback für fehlende Features).
- **[Risiken]** Modus-Fallback ist still — wenn das LLM nur 2 Features spezifiziert, kommen 7 Features aus dem Test-Modus, ohne dass das LLM es weiß. Bewusster Kompromiss für Robustheit.

### Block: Zeilen 274–318 — `_tool_get_shap_values(instance_id)`
- Liefert vollständige lokale Erklärung: `prediction`, `base_value`, `contribution_space="log"`, `contributions` (sortiert), `feature_values`, plus eine `"note"`, die dem LLM **explizit** die multiplikative `exp()`-Mechanik erklärt — sehr saubere Prompt-Engineering-Entscheidung.

### Block: Zeilen 320–351 — `_tool_get_partial_dependence`
- PD klassisch: pro Grid-Punkt **alle X_test** auf den Wert setzen, Mittelwert der Vorhersagen.
- **[Kniff Zeile 338-340]** Nach `X_copy[feature] = val` muss bei kategorischen Spalten der `category`-Dtype wiederhergestellt werden, sonst bricht `enable_categorical=True`.
- **[Risiken]** `np.linspace(min,max,20)` auf Testset → mögliche Out-of-Distribution-Werte am Rand.

### Block: Zeilen 353–385 — `_tool_get_feature_value_context`
- Numerisch: Percentile + min/max/mean/std.
- Kategorisch: Counts + Prozentverteilung.
- Damit kann das LLM Aussagen wie „27 °C im Mai liegt im 78. Perzentil" formulieren.

### Block: Zeilen 387–433 — `_tool_get_similar_instances`
- Min-Max-normierte euklidische Distanz auf `cat.codes`-kodierten Features.
- **[Toter Code Zeile 416]** `y_train = self.X_train.copy()` wird angelegt und nie genutzt.
- **[Risiken]** `weathersit` (nominal) bekommt durch `.cat.codes` quasi-ordinale Distanz. Für KNN-Plausibilität OK, semantisch nicht ideal.

### Block: Zeilen 435–469 — `_tool_get_counterfactual_prediction`
- Klont Test-Instanz, überschreibt nur `changes`, gibt `delta = cf_pred - base_pred` zurück.
- **[Defensive Zeile 450-453]** Kategoriewerte durch `pd.Categorical([...], categories=...)` erzwungen → fremde Werte werden zu NaN (nicht abgefangen — eine Verbesserungsmöglichkeit).

### Block: Zeilen 480–495 — `_build_input_df`
- Baut DataFrame mit korrekten Dtypes; fehlende Features → `mode()` aus X_test.

### TL;DR
**Die agentische Werkzeugkiste** — robust, dtype-sicher, mit minimalen Lücken (Tot-Code, ein Drift-Hint im Schema).

---

## Notebook: [01_Data_Preprocessing.ipynb](notebooks/01_Data_Preprocessing.ipynb)

**Zweck:** UCI Bikesharing `hour.csv` → bereinigtes `train.csv`/`test.csv`.

### Block: Zelle 1 — Setup
```python
import sys, numpy, pandas, matplotlib, seaborn
sys.path.insert(0, str(Path.cwd().parent))
from utils import DATA_DIR, RANDOM_STATE
DATA_PATH = DATA_DIR / "hour.csv"
```
- **[Originalfunktion]** Lädt Bibliotheken, fügt Projekt-Root in `sys.path` ein (damit `from utils import ...` aus dem `notebooks/`-Unterordner funktioniert), referenziert `DATA_DIR`.
- **[Risiken]** `sys.path.insert(0, ...)` ist fragil bei Notebook-Restart aus anderem CWD, aber Standard für Forschungs-Notebooks.

### Block: Zellen 2–6 — Rohdatensicht
```python
df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
df.info(); df.describe(); df.isna().sum(); df.duplicated().sum()
expected_hours = pd.date_range(start=df["dteday"].min(), end=..., freq="h")
```
- **[Originalfunktion]** Rohinspektion: Shape, dtypes, Missings, Duplikate, Stundenlücken-Prüfung.
- **[Semantik]** Validiert die Annahme „eine Zeile pro Stunde". Aus dem Datenblatt der UCI-Quelle bekannt, dass es **nicht** lückenlos ist — der Vergleich liefert die exakte Anzahl fehlender Stunden.

### Block: Zelle 7 — Leakage-Entfernung
```python
leakage_cols = ["casual", "registered"]
id_cols = ["instant"]
df = df.drop(columns=leakage_cols + id_cols)
```
- **[Originalfunktion]** Entfernt `casual` + `registered` (deren Summe ist `cnt` → Target-Leakage) und `instant` (Zeilen-ID).
- **[Warum hier?]** Vor jeder weiteren Analyse, damit nachfolgende Korrelations- und Skewness-Plots auf der echten Feature-Menge laufen.

### Block: Zelle 8 — Skewness-Inspektion
- Two-panel-Histogramm `cnt` vs `log1p(cnt)`. Print von `skew()`-Werten.
- **[Semantik]** Begründet datentechnisch, warum später Poisson-Loss (log-Link) sinnvoll ist: roh ≈ 2,44 → log1p ≈ 0,17 (von rechtsschief zu nahezu symmetrisch).

### Block: Zelle 9 — `cnt_log1p` Spalte hinzufügen
```python
df["cnt_log1p"] = np.log1p(df["cnt"])
```
- **[Wichtig]** Diese Spalte wird in `train.csv` mit gespeichert, aber `utils.data.DROP_COLS` wirft sie beim Laden wieder raus. Konsequenz: 02a kann nicht auf log1p direkt trainieren via `load_train_test()` — die XGBoost/EBM-Modelle nutzen stattdessen native Poisson-Objectives auf rohem `cnt`. Das ist die saubere Lösung.

### Block: Zelle 10 — Redundanz & Dtype-Setup
```python
df = df.drop(columns=["workingday", "season"])
categorical_cols = ["mnth", "hr", "weekday", "weathersit"]
for col in categorical_cols:
    df[col] = df[col].astype("category")
```
- **[Originalfunktion]** Entfernt redundante Features und markiert kategorische Spalten.
- **[Semantik]** Begründung im Kommentar:
  - `workingday = NOT(holiday OR weekday∈{0,6})` → vollständig ableitbar
  - `season ≈ floor((mnth-3)/3) mod 4` → vollständig ableitbar
- **[Risiken]** `category`-Dtypes überleben den `to_csv`-Roundtrip nicht — `utils.data._apply_dtypes` muss sie beim Lesen wiederherstellen.

### Block: Zelle 11–12 — Korrelations-Check & `atemp`-Drop
```python
corr = df[numeric_cols].corr(); sns.heatmap(...)
df = df.drop(columns=["atemp"])
```
- Visualisiert die r ≈ 0,99-Korrelation von `temp`/`atemp` und droppt eine.

### Block: Zelle 13 — Split
```python
train_df, test_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
```
- **[Wichtig]** **Zufälliger** Split (kein zeitlicher), explizit begründet: zeitlicher Split (2011 train / 2012 test) hätte die `yr`-Variable verzerrt, da 2012 ≈ 57 % mehr Ausleihen hatte.
- **[Risiken]** Bei Zeitreihendaten ist random-split **theoretisch problematisch** (potenzielle temporale Leakage benachbarter Stunden). Notebook 07 listet das als methodische Einschränkung #6. R² > 0,95 ist teilweise dadurch erklärbar — für eine XAI-Demonstration vertretbar.

### Block: Zellen 14–15 — Speichern
```python
train_df.to_csv(OUT_DIR / "train.csv", index=False)
test_df.to_csv(OUT_DIR / "test.csv", index=False)
```

### TL;DR Notebook 01
Klassisches Preprocessing mit guter Begründung jedes Schritts; einzige nicht-triviale Designentscheidung ist der **zufällige** Split — bewusst gewählt, methodisch markiert.

---

## Notebook: [02a_Modeling_AllOptions.ipynb](notebooks/02a_Modeling_AllOptions.ipynb)

**Zweck:** Trainiert XGBoost + EBM für alle drei Loss-Optionen in **einer Schleife** und speichert Metriken + Modelle.

### Block: Zellen 1–2 — Setup + Datenladen
```python
from utils.data import load_train_test
from utils.models import LOSS_OPTIONS, compute_metrics, save_model
X_train, y_train, X_test, y_test = load_train_test()
```
- **[Wichtig]** Hier kommt `load_train_test` zum ersten Mal real zum Einsatz — `category`-Dtypes werden rekonstruiert, ohne dass das Notebook etwas davon merkt.

### Block: Zelle 3 — Skewness-Plot
- Wiederholt die Verteilungsvisualisierung aus Notebook 01 zur Selbstkontrolle.

### Block: Zelle 4 — Hyperparameter
```python
xgb_params = dict(n_estimators=800, max_depth=7, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                  reg_lambda=1.0, enable_categorical=True, tree_method="hist",
                  n_jobs=-1, random_state=RANDOM_STATE)
ebm_params = dict(interactions=15, max_bins=512, learning_rate=0.02,
                  max_rounds=10000, early_stopping_rounds=100,
                  n_jobs=-1, random_state=RANDOM_STATE)
```
- **[Wichtig]** **Identische** Parameter über alle drei Loss-Optionen → fair vergleichbar. Einziger Unterschied ist `objective`.
- **[Semantik]**
  - `enable_categorical=True` aktiviert die XGBoost-native Behandlung von `category`-Dtype (erfordert XGBoost ≥ 1.6).
  - `tree_method="hist"` ist schneller und unterstützt Categorical-Splits.
  - EBM `interactions=15` schaltet 15 Paar-Interaktionsterme zusätzlich zu den Haupteffekten ein.

### Block: Zelle 5 — Trainingsschleife
```python
for LOSS in LOSS_OPTIONS.values():
    xgb = XGBRegressor(objective=LOSS.xgb_objective, **xgb_params)
    xgb.fit(X_train, y_train)
    ebm = ExplainableBoostingRegressor(objective=LOSS.ebm_objective, **ebm_params)
    ebm.fit(X_train, y_train)
    # metrics + save_model + JSON-Dump
```
- **[Originalfunktion]** Sechs Modelle (2 × 3) trainieren, Metriken berechnen, je `xgb_<loss>.pkl`/`ebm_<loss>.pkl` + `model_metrics_<loss>.json` speichern.
- **[Semantik]** **Option 2 und 3 trainieren _denselben_ Loss** (beide `count:poisson` / `poisson_deviance`) → die Modelle sind numerisch identisch. Der Unterschied entsteht erst in 03 bei der Beitragsextraktion. Die Schleife trainiert sie aus didaktischer Klarheit trotzdem doppelt.
- **[Risiken]** Doppeltraining ist Verschwendung (~2 × CPU-Zeit). Effizienter wäre: Option 2 + 3 teilen sich das `.pkl` und unterscheiden sich nur im Beitragsextraktor-Flag. Akzeptabel für Belegarbeit.

### Block: Zelle 6 — Ergebnistabelle + Sanity-Check
```python
diff = max(abs(m2[k] - m3[k]) for k in ("rmse","mae","r2"))
status = "✓ identisch" if diff < 1e-9 else f"⚠ Δ={diff:.2e}"
```
- **[Originalfunktion]** Validiert, dass Option 2 und 3 numerisch identisch sind. Wenn ja → Bestätigung, dass die Loss-Wahl korrekt implementiert ist.

### Block: Zelle 7 — Diagnostik Plots
- Scatter `y_true` vs `y_pred` pro Option × Modell. Warnung wenn negative Vorhersagen vorkommen — der visuell stärkste Befund für Option 1 (Squared Error) bei Count-Daten.

### TL;DR Notebook 02a
Eine einzige Schleife trainiert alles, speichert alles, validiert die Loss-Identität. Sehr sauber.

---

## Notebook: [02b_Comparison.ipynb](notebooks/02b_Comparison.ipynb)

**Zweck:** Konsolidiert die 3 Metriken-JSONs aus 02a, erstellt eine Tabelle, validiert Option-2/3-Identität, beantwortet die Modell-Ebene-Frage.

### Block: Zelle 1 — JSON-Laden
```python
for key in LOSS_OPTIONS:
    path = RESULTS_DIR / f"model_metrics_{key}.json"
    if path.exists(): results[key] = json.loads(path.read_text())
    else: print(f"WARNUNG: {path} fehlt — Notebook 02{'abc'[idx]} noch nicht gelaufen?")
```
- **[Originalfunktion]** Defensives Laden mit informativer Warnung.
- **[Stolperfalle]** Die Warnung referenziert "02abc" — Reminiszenz an ein früheres Split-Layout, das jetzt zu 02a + 02b konsolidiert wurde. Kosmetisch veraltet, harmlos.

### Block: Zellen 2–3 — Konsolidierte Tabelle + Δ-Check
- Pivotiert die Metriken zu einer Tabelle (Option × Modell × Metrik).
- Quantifiziert Differenz Option 2 vs Option 3:
```python
max_abs_diff = diff_df["|Differenz|"].max()
if max_abs_diff < 1e-6: print("=> Option 2 und 3 IDENTISCH")
```

### Block: Zelle 4 — Visualisierung
- 2×2-Grid: RMSE, MAE, R², Poisson-Dev. pro Option, gruppiert nach Modell. „↑ besser" / „↓ besser"-Annotation pro Subplot.

### Block: Zelle 5 — Negative-Predictions-Plausibilität
- Tabelle `min(pred)` und `neg. preds` pro Option. **Der praktische Killer-Befund** für Option 1: 108 (XGB) / 358 (EBM) negative Fahrradzahlen — domänenfremd, würde LLMs verwirren.

### Block: Zelle 6 — Zusammenfassung
- Pro Modell: beste Option je Metrik. Hartkodierte Hinweise:
  - „Option 2 und 3 auf Modell-Ebene identisch."
  - „Option 1 → Plausibilitäts-Risiko."
  - „Final-Entscheidung braucht zusätzlich Erklärungs-Faithfulness (NB 07)."

### Block: Zelle 7 — CSV-Export
- `results/model_comparison_summary.csv`.

### TL;DR Notebook 02b
Reine Aggregations-/Visualisierungs-Schicht. Liefert das Argument für „Poisson-Log gewinnt".

---

## Notebook: [03_Explanations_Generation.ipynb](notebooks/03_Explanations_Generation.ipynb)

**Zweck:** Generiert globale + lokale SHAP/EBM-Erklärungen für **Option 2 (poisson_log)** und speichert sie als JSON.

### Block: Zellen 1–2 — Setup + Laden
```python
LOSS_KEY = "poisson_log"
xgb = joblib.load(MODELS_DIR / f"xgb_{LOSS_KEY}.pkl")
ebm = joblib.load(MODELS_DIR / f"ebm_{LOSS_KEY}.pkl")
metrics = json.loads((RESULTS_DIR / f"model_metrics_{LOSS_KEY}.json").read_text())["metrics"]
```
- **[Wichtig]** Hier wird die Option-2-Entscheidung **operationalisiert**: ab hier laufen alle Pipelines nur noch auf dem Poisson-Log-Modell.

### Block: Zellen 3–4 — Globale Erklärungen
```python
global_xgb = build_global(xgb, "xgb", X_train, metrics["xgb"])
save_explanation(global_xgb, f"global_xgb_{LOSS_KEY}.json")
```
- Ruft direkt `utils.explanations.build_global` auf (SHAP-Cache wird hier zum ersten Mal aufgebaut).
- Druckt Top-5 Features zur visuellen Bestätigung.

### Block: Zellen 5–6 — Lokale Erklärungen für die 10 Instanzen
```python
for iid in INSTANCE_IDS:
    local = build_local(xgb, "xgb", X_test, y_test, iid)
    save_explanation(local, f"local_xgb_{LOSS_KEY}_inst{iid}.json")
```
- 10 × 2 = **20 JSONs** erzeugt: `local_{xgb,ebm}_poisson_log_inst{ID}.json`.

### Block: Zellen 7–8 — Übersicht + Sanity-Check
- Listet die Datei-Größen.
- Vergleicht Rang-Übereinstimmung XGB vs EBM global. Aus dem Readme bekannt: beide haben `hr` als Top-1; danach divergieren sie leicht.

### TL;DR Notebook 03
Reine I/O-Schicht. Liefert die 22 JSONs (2 global + 20 local), die die LLM-Pipelines lesen.

---

## Notebook: [04_LLM_JSON_Pipeline.ipynb](notebooks/04_LLM_JSON_Pipeline.ipynb)

**Zweck:** Pipeline 1 — JSON → Text. Pro (Modell, Instanz) wird eine deutsche Erklärung erzeugt und in `results/pipeline04/` gespeichert.

### Block: Zelle 1 — Konfig
```python
from utils.llm import ask_text, DEFAULT_MODEL
LOSS_KEY = "poisson_log"; MODEL = DEFAULT_MODEL; MAX_TOKENS = 600
```

### Block: Zelle 2 — `SYSTEM_PROMPT`
Ein ≈ 2000-Zeichen-Block mit drei klar markierten Sektionen:
1. **`=== DOMAIN-KONTEXT ===`** — Bikesharing in DC + Poisson-Log-Mechanik (`exp(Basiswert + Σ Beiträge)`).
2. **`=== FEATURE-SCHEMA ===`** — jedes der 9 Features mit Bedeutung + Verhaltens-Hinweisen (z. B. „temp 0.5–0.8 = 20–33 °C, optimal").
3. **`=== AUSGABEFORMAT ===`** — Pflicht zu `[VORHERSAGE]`/`[TREIBER]`/`[EMPFEHLUNG]`, 150–250 Wörter, kein SHAP-Fachjargon.
- **[Warum so lang?]** Der System-Prompt muss > 1024 Tokens haben, damit Anthropic Prompt Caching greift. Die ausführlichen Feature-Beschreibungen erfüllen diese Schwelle zuverlässig **und** machen das LLM domänen-kalibriert.
- Print am Ende: Längen-Check (Zeichen + geschätzte Tokens).

### Block: Zelle 3 — Hilfsfunktionen
```python
WEEKDAYS = ["Sonntag",..."Samstag"]
MONTHS = ["", "Januar",...,"Dezember"]
WEATHER = {1:"klar/wenige Wolken",...,4:"Starkregen/Gewitter"}

def build_context_string(fv): ...
def load_explanations(model_name, instance_id) -> tuple[dict,dict]: ...
def build_user_prompt(global_exp, local_exp, top_k=5) -> str: ...
```
- **[Originalfunktion]**
  - `build_context_string`: Denormalisierung (`temp*41`, `hum*100`, `windspeed*67`) zu Alltagssprache.
  - `load_explanations`: Liest `global_*.json` + `local_*.json` aus 03.
  - `build_user_prompt`: Baut Mini-Payload mit Metriken (rmse/r²/poisson_dev), Top-5-Global-Features, Feature-Werten, `human_readable_context`-String, y_true/prediction, Top-6-Beiträgen.
- **[Designentscheidung]** **Pre-Denormalisierung in der Pipeline** (statt das LLM denormalisieren zu lassen) → reduziert Halluzinations-Risiko.

### Block: Zelle 4 — LLM-Schleife
```python
for model_name in ["xgb", "ebm"]:
    for iid in INSTANCE_IDS:
        response = ask_text(user_prompt, system=SYSTEM_PROMPT, cache_system=True, ...)
        # extract text, usage, store record
```
- 20 API-Calls (2 × 10).
- Loggt input/output/cache-tokens und Latenz pro Call.
- **[Wichtig]** Erster Call füllt den System-Prompt-Cache, alle weiteren 19 lesen daraus (`cache_read_input_tokens`).

### Block: Zellen 5–6 — Ausgabe
- Zeigt 2 Beispiel-Erklärungen voll ausgedruckt.
- Tabelle mit Wörtern/Tokens/Zeit.

### TL;DR Notebook 04
Hartcodierter, präzise kalibrierter System-Prompt + denormalisierte JSON-Payloads. Effizient durch Prompt-Caching.

---

## Notebook: [05_LLM_Vision_Pipeline.ipynb](notebooks/05_LLM_Vision_Pipeline.ipynb)

**Zweck:** Pipeline 2 — LLM sieht den Waterfall-Plot als Bild und erklärt ihn.

### Block: Zelle 1 — Setup
```python
matplotlib.use('Agg')  # headless backend für Plot-Generierung ohne Display
from utils.llm import ask_with_images, DEFAULT_MODEL
```
- **[Wichtig]** `Agg` ist essentiell für saubere Speicherung in nicht-interaktiven Kontexten.

### Block: Zelle 2 — Modelle + SHAP einmal
```python
shap_explainer = shap.TreeExplainer(xgb)
shap_values    = shap_explainer(X_test)
```
- SHAP über das gesamte Test-Set vorberechnet → Plot-Generierung pro Instanz wird billig.

### Block: Zelle 3 — `plot_xgb_waterfall(instance_id)`
```python
pos = list(X_test.index).index(instance_id)
shap.plots.waterfall(shap_values[pos], show=False, max_display=10)
plt.title(f'XGBoost – Instanz {instance_id} | Vorhersage: {pred:.1f} (tatsächlich: {y:.0f})')
plt.savefig(path, dpi=130, bbox_inches='tight'); plt.close('all')
```
- **[Subtilität]** `list(X_test.index).index(instance_id)` ist O(n) und nimmt an, dass `instance_id` der **Pandas-Index** ist (nicht iloc-Position). Das **steht im Widerspruch** zur Vereinbarung in `utils.explanations.build_local`, die `iloc` nutzt. Da `X_test` nach dem `train_test_split` und `reset_index(drop=True)` in 01 einen 0..N-1-Index hat, sind iloc und Index hier identisch — funktioniert, ist aber fragil. **Wahrscheinlich** kein realer Bug, aber bei späterem Refactor gefährlich.
- `shap.plots.waterfall` ist die offizielle SHAP-API; benötigt `Explanation`-Objekt (deshalb `shap_explainer(X_test)` statt `shap_values()`).

### Block: Zelle 3 — `plot_ebm_waterfall(instance_id)`
```python
exp = ebm.explain_local(inst); d = exp.data(0)
base = float(d['extra']['scores'][0])
pairs = [(n, float(s)) for n, s in zip(d['names'], d['scores'])]
pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:10]
colors = ['#d73027' if v > 0 else '#4575b4' for v in scores]
ax.barh(labels[::-1], scores[::-1], color=colors[::-1], height=0.65)
```
- **[Originalfunktion]** Manueller Waterfall, weil `interpret` Plotly liefert (nicht Matplotlib).
- Rot für positiv, Blau für negativ — bewusst SHAP-konsistente Farbcodierung.
- `[::-1]` dreht für `barh` (von oben nach unten).

### Block: Zelle 4 — Plot-Generierung
```python
for iid in INSTANCE_IDS:
    p_xgb = plot_xgb_waterfall(iid)
    p_ebm = plot_ebm_waterfall(iid)
    plot_paths[('xgb', iid)] = p_xgb
    plot_paths[('ebm', iid)] = p_ebm
```
- 20 PNGs erzeugt in `explanations/plots/`.

### Block: Zelle 5 — Beispielanzeige
- Nutzt `IPython.display.Image` — nur in Jupyter sichtbar, in Headless-Runs harmlos.

### Block: Zelle 6 — `SYSTEM_PROMPT`
Eigener Prompt mit Sektion **`=== WATERFALL-PLOT LESEN ===`**, die dem LLM erklärt:
- Roter Balken = erhöht; Blauer Balken = senkt.
- `E[f(X)]` / base value = Log-Raum-Ausgangspunkt.
- `exp(f(x)) ≈ Ausleihen`.
- Sortierung nach |Einfluss|.
- Feature-Wert steht beim Namen.

→ Das ist die **didaktische Übersetzung der Plot-Konventionen** — ohne diesen Prompt würde das LLM raten.

### Block: Zelle 7 — `build_context_prompt(model_name, instance_id)`
- Erzeugt strukturierten Text mit **lesbaren Feature-Werten + y_true + prediction**, der **zusätzlich zum Bild** geschickt wird.
- **[Designentscheidung]** Bild + redundanter Text → Vision-Pipeline ist nicht „rein visuell", sie ist „Vision + Kontext". Macht das LLM robuster gegen Bild-Ablese-Fehler bei kleinen Balken.

### Block: Zelle 7 — LLM-Schleife
```python
response = ask_with_images(prompt, image_paths=[plot_path], system=SYSTEM_PROMPT, ...)
```
- 20 multimodale Calls.

### Block: Zellen 8–9 — Beispiele + Tabelle
- Wie Notebook 04.

### TL;DR Notebook 05
**„Vision + Kontext"** statt rein visuell. Praktischer Kompromiss zwischen wissenschaftlicher Reinheit und Erklärungsqualität.

---

## Notebook: [06_LLM_ToolUse_Pipeline.ipynb](notebooks/06_LLM_ToolUse_Pipeline.ipynb)

**Zweck:** Pipeline 3 — Agentic loop. Das LLM fragt Daten aktiv über Tools ab.

### Block: Zellen 1–2 — Setup + Laden
```python
from utils.tools import ToolBox, TOOL_DEFINITIONS
from utils.llm import DEFAULT_MODEL, _get_client, _with_retry
MAX_TOKENS = 1500  # höher als in 04/05 (1500 vs 600), weil mehr Output erwartet
```
- **[Wichtig]** `_get_client` und `_with_retry` sind **Underscore**-Imports — eigentlich „private". Pragmatisch, weil die Tool-Use-Schleife eigene Logik braucht (nicht in `ask_text` einbaubar).

### Block: Zelle 3 — `SYSTEM_PROMPT` mit Tool-Anleitung
- Drei Sektionen:
  1. `=== KONTEXT ===` mit `{model_name}`-Platzhalter (wird pro Call gefüllt).
  2. `=== EMPFOHLENE TOOL-REIHENFOLGE ===` mit konkreter 6-Schritt-Anleitung:
     1. `get_shap_values(iid)` → lokale Treiber
     2. `get_feature_importance()` → global zum Vergleich
     3. `get_feature_value_context(iid, feature)` für Top-2-Treiber → typisch?
     4. `get_counterfactual_prediction(iid, changes)` → Was-wäre-wenn
     5. (optional) `get_partial_dependence(feature)`
     6. (optional) `get_similar_instances(iid)`
  3. `=== AUSGABEPFLICHT ===` + `=== AUSGABEFORMAT ===` (200–300 Wörter, dreiteilig).
- **[Wichtig]** Die explizite Reihenfolge ist **Prompt-Engineering** — ohne sie würde das LLM willkürlich/zu wenig Tools rufen. Mit ihr werden im Schnitt ≈ 5,85 Calls erreicht (laut [Readme.md:166](Readme.md:166)).

### Block: Zelle 3 — `tool_use_loop`
```python
def tool_use_loop(client, toolbox, user_message, system, max_rounds=10):
    messages = [{"role": "user", "content": user_message}]
    for round_num in range(max_rounds):
        response = _with_retry(client.messages.create,
            model=MODEL, max_tokens=MAX_TOKENS,
            system=[{"type":"text","text":system,"cache_control":{"type":"ephemeral"}}],
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        # cache last assistant text in case end_turn comes after tool_use
        candidate = next((b.text for b in response.content if hasattr(b,"text") and b.text), "")
        if candidate: last_text = candidate
        if response.stop_reason == "end_turn":
            return last_text, toolbox.call_log, total_in, total_out
        if response.stop_reason != "tool_use":
            break
        tool_results = []
        for block in response.content:
            if block.type != "tool_use": continue
            result = toolbox.dispatch(block.name, block.input)
            tool_results.append({"type":"tool_result","tool_use_id":block.id,
                                 "content":json.dumps(result, ensure_ascii=False)})
        messages.append({"role":"user","content":tool_results})
```

**Im Detail:**
- **[Originalfunktion]** Klassische Anthropic-Tool-Use-Schleife mit `stop_reason`-Handling.
- **[Pattern]** `messages`-Liste wächst: User-Prompt → Assistant-(text+tool_use)-Block → User-(tool_results)-Block → Assistant → … bis `end_turn`.
- **[Wichtiger Trick `last_text`]** Wenn der Assistant gleichzeitig Text **und** Tool-Use ausgibt (was Claude bei `stop_reason="tool_use"` machen kann), wird der Text gespeichert. Falls die letzte Runde nur `tool_use` ohne Text liefert oder `max_rounds` reicht, wird der letzte gespeicherte Text zurückgegeben — verhindert leere Erklärungen.
- **[Prompt-Caching auch hier]** Der System-Prompt wird mit `cache_control: ephemeral` markiert. Bei 20 Calls (10 pro Modell × 2 Modelle) ergibt das massive Einsparungen — der Cache ist allerdings **kurzlebig** (5 min TTL), realistisch greift er nur, wenn die Calls sequenziell schnell genug laufen.
- **[Risiken]**
  - `tool_results.append(...)` ohne `tool_use_id` würde von Anthropic gerejected — die Implementierung übergibt es korrekt.
  - Falls eine `dispatch`-Exception in `ToolBox` als `{"error": ...}` zurückkommt, sieht das LLM nur den String — keine HTTP-Fehler, kein Crash.
  - `break` bei unbekanntem `stop_reason` (z. B. `max_tokens`) gibt den letzten Teiltext zurück — pragmatisch, aber maskiert potenziell echte Probleme.
- **[Einfach erklärt]** „Schicke dem LLM die Frage, gib ihm Tools, lass es ein paar Mal nachfragen, gib am Ende seinen letzten Text zurück."

### Block: Zelle 4 — `build_task_prompt(model_name, instance_id)`
- Wie in 05: lesbarer Kontext (Uhrzeit, Wochentag, Temperatur in °C, Wetterlage etc.).
- Schließt mit der Anweisung **„Bitte nutze mindestens 4 der verfügbaren Tools"** — zweite Reminder zusätzlich zum System-Prompt, weil LLM-Compliance mit Tool-Aufrufen sonst unzuverlässig ist.

### Block: Zelle 5 — Aufrufe
```python
client = _get_client()
for model_name, model_obj in [('xgb', xgb), ('ebm', ebm)]:
    for iid in INSTANCE_IDS:
        toolbox = ToolBox(model_obj, X_train, X_test, y_test, model_name)
        prompt  = build_task_prompt(model_name, iid)
        system  = SYSTEM_PROMPT.replace('{model_name}', model_name.upper())
        text, call_log, in_tok, out_tok = tool_use_loop(client, toolbox, prompt, system)
```
- **[Wichtig]** Pro Instanz wird eine **neue** `ToolBox` erzeugt → frischer `call_log`. Das ist absichtlich, damit das LLM in jedem Run unbeeinflusst von vorigen Aufrufen entscheidet.
- Speichert `n_tool_calls`, `tool_calls`-Trace, Token-Usage, Latenz → Goldmine für Notebook 07.

### Block: Zellen 6–7 — Beispiel + Zusammenfassung
- Erste XGB-Erklärung wird voll ausgedruckt **plus** der Tool-Trace (Tool-Name + Args + Result-Preview).

### TL;DR Notebook 06
Saubere Anthropic-Tool-Use-Implementierung mit `stop_reason`-Loop, `last_text`-Fallback, Prompt-Caching. Komplexester Notebook der Suite.

---

## Notebook: [07_Evaluation.ipynb](notebooks/07_Evaluation.ipynb)

**Zweck:** Bewertet die drei Pipelines auf drei Ebenen: quantitativ, Keyword-Faithfulness, LLM-as-Judge (drei Versionen).

### Block: Zelle 1 — Konfig + Preise
```python
COST_INPUT_PER_M       = 3.00   # USD/M reguläre Input-Tokens
COST_CACHE_READ_PER_M  = 0.30   # USD/M Cache-Read (90% Rabatt)
COST_OUTPUT_PER_M      = 15.00  # USD/M Output
```
- **[Wichtig]** Cache-Read-Preis ist explizit modelliert → für Pipeline-Vergleich relevant, weil 04/05 starkes Caching haben, 06 wegen multi-turn weniger.

### Block: Zelle 2 — Ergebnisse laden + Kosten berechnen
```python
regular_in = max(in_tok - cache_r, 0)
cost = (regular_in*COST_INPUT_PER_M + cache_r*COST_CACHE_READ_PER_M + out_tok*COST_OUTPUT_PER_M) / 1e6
```
- **[Originalfunktion]** Rechnet exakte USD-Kosten pro Call.
- **[Subtilität]** Anthropic-Usage liefert `input_tokens` (inkl. cached) und `cache_read_input_tokens` separat — der **reguläre Anteil** ist die Differenz. Cache-Tokens sind 10× billiger.

### Block: Zelle 3 — Instance-Doku
- Hardcoded `instance_doc` mit den 10 IDs, deren echten `cnt`-Werten und Kontext-Notizen. Reine Reproduzierbarkeits-Tabelle.

### Block: Zelle 4 — Quantitative Aggregation
- Aggregiert pro Pipeline: Wörter, Tokens (Input/Output/Total), Gesamt-USD, Ø-Zeit, Ø-Tool-Calls.

### Block: Zelle 5 — Visualisierung
- 3-Spalten-Subplot: Wortzahl, Token-Verbrauch (stacked Input/Output), Gesamtkosten. Speichert `eval_quantitative.png`.

### Block: Zelle 6 — Faithfulness keyword-basiert
```python
FEATURE_KEYWORDS = {
    "hr": ["uhrzeit","stunde","tageszeit","uhr","morgen","abend","nacht","mittag"],
    "temp": ["temperatur","wärme","warm","kalt","grad","°c"],
    ...
}
def mentions_feature(text, feature):
    return any(kw in text.lower() for kw in FEATURE_KEYWORDS.get(feature,[feature]))
```
- **[Originalfunktion]** Prüft pro Top-3-Feature, ob mindestens ein deutsches Synonym im Erklärungstext vorkommt.
- **[Schwäche]** Keyword-Match ist crude — z. B. „kein Feiertag" → matched `holiday` als positiv, obwohl es als „nicht relevant" abgewertet wurde. Aber: konsistente Schwäche über alle Pipelines → vergleichbar.

### Block: Zellen 7–8 — LLM-as-Judge v1 (unkalibriert)
- `JUDGE_SYSTEM` mit 3-Kriterien-Rubrik (Faithfulness/Clarity/Completeness, je 1–5).
- Wichtig: Im Judge-Prompt sind die **echten** Top-3 + denormalisierte Feature-Werte als `ground_truth` enthalten → Judge kann prüfen.
- **Tool-Use-Special** (Zeile 336-343): Wenn die Pipeline 06 ist, wird ein zusätzlicher `tool_context`-Hinweis in den Judge-Prompt gepackt, der dem Judge erklärt, dass Zahlen aus Tool-Calls korrekt sind, auch wenn nicht in den Top-3 stehen. → Versuch, den strukturellen Nachteil von Tool-Use beim Judge auszugleichen.
- 30 Calls (3 Pipelines × 2 Modelle × 5 Instanzen — Anmerkung: das Notebook schreibt zwar 30, aber 3×2×10=60 wäre die korrekte Anzahl; **wahrscheinlich** wurden in einer früheren Version nur 5 Instanzen gefahren, die Zahl wurde nicht aktualisiert — `eval_llm_judge.json` enthält die tatsächliche Anzahl).
- Speichert in `eval_llm_judge.json`.

### Block: Zellen 9–10 — Aggregation + Radar-Plot
- Ø-Scores pro Pipeline; Radar-Chart mit 3 Achsen (Faithfulness/Clarity/Completeness).
- Speichert `eval_radar.png`.

### Block: Zelle 11 — Empfehlung
```python
judge_total = judge_df.groupby('pipeline_label')[['faithfulness','clarity','completeness']].mean().sum(axis=1)
best_pipeline = judge_total.idxmax()
```
- Print mit Trade-off-Analyse für jede Pipeline.

### Block: Zelle 13 — Judge v2 (kalibrierter Prompt) laden
```python
v2_path = RESULTS_DIR / 'eval_llm_judge_v2.json'
```
- Lädt nur aus Cache (wurde extern erzeugt, evtl. mit anderem Prompt).
- Wichtigste Statistik: **Ceiling-Effekt** quantifizieren — wie viele Scores = 5? v1: 91 %. v2: 73 %.

### Block: Zelle 14 — v1 vs v2 Faithfulness-Vergleich
- Bar-Chart links: v1 vs v2 pro Pipeline.
- Bar-Chart rechts: Score-Verteilung 1–5 für v2.
- **Befund:** v2 ist strenger; Tool-Use wird in v2 fairer bewertet, weil die Rubrik explizit „abgerufene Zahlen sind Beleg" zulässt.

### Block: Zelle 15 — Judge v3 (Opus 4.7, unabhängig)
- Lädt `eval_llm_judge_opus.json`.
- **Motivation:** Self-Preference-Bias eliminieren — v1/v2 nutzten Sonnet sowohl als Generator als auch als Judge. Opus ist unabhängig.
- **Wichtige Caveat im Notebook-Kommentar:** Opus sieht die Tool-Call-Outputs nicht → Zahlen aus PD-Kurven / Counterfactuals gelten als nicht-belegbar → strukturell niedrigere Tool-Use-Scores. Dokumentations-Highlight, dass das **Messartefakt** ist, kein Qualitätsmangel.

### Block: Zelle 16 — Dreifach-Vergleich
- 3 Subplots (Faithfulness/Clarity/Completeness), je 3 Balken (v1/v2/Opus) pro Pipeline.
- Speichert `eval_judge_all_versions.png`.
- Print mit Kernbefunden: Completeness robust (alle ≈ 5); Faithfulness divergiert je nach Judge-Strenge; Vision konsistent ≈ 4,4.

### Block: Zelle 17 — Methodische Einschränkungen
- Tabelle mit 8 dokumentierten Einschränkungen (Ceiling, Self-Preference, n=10, kein Repeated Sampling, Convenience-Sampling, Random-Split bei Zeitreihen, Selection-Bias der IM-Metriken, unkalibrierter Judge). Wissenschaftlich vorbildlich.

### TL;DR Notebook 07
Sehr gründliche Multi-Judge-Evaluation. Wissenschaftlicher Höhepunkt der Implementierung.

---

## Notebook: [08_Evaluation_Ichmoukhamedov.ipynb](notebooks/08_Evaluation_Ichmoukhamedov.ipynb)

**Zweck:** Implementiert das formale Faithfulness-Framework aus Ichmoukhamedov et al. (2024) — RA/SA/VA-Metriken plus Assumptions-Extraktion.

### Block: Zellen 1–2 — Setup + Datenladen
```python
TOP_K = 4  # Paper: top-4 features
PIPELINES = ['04','05','06']
```
- **[Wichtig]** Pro Narrativ wird die zugehörige `local_*.json`-Ground-Truth-Erklärung mitgeladen → für jede der 60 (3×2×10) Erklärungen ist die SHAP-Wahrheit verfügbar.

### Block: Zelle 3 — `EXTRACTION_SYSTEM` + `build_extraction_prompt`
- Zweistufiges System: Generator-LLM → Extraction-LLM.
- Das Extraction-LLM (auch Claude Sonnet) bekommt das Narrativ als Input und soll daraus pro Feature extrahieren: `rank`, `sign (+1/-1)`, `value`, `assumption`.
- Beispiel-Ausgabeformat ist im Prompt mitgeliefert (Few-shot).
- **[Schlauer Trick `_DENORM`]** `{'temp': lambda v: v*41, 'hum': v*100, 'windspeed': v*67}` — Lookup für Wert-Vergleich auf beiden Skalen.

### Block: Zelle 4 — Extraktionen mit Cache
```python
CACHE_PATH = OUT_DIR / 'extractions.json'
if CACHE_PATH.exists():
    extractions = json.loads(CACHE_PATH.read_text())
```
- **[Originalfunktion]** Idempotente Ausführung: bereits berechnete Extraktionen werden aus Cache geladen, nur fehlende API-Calls gemacht.
- 60 mögliche Calls (3 × 2 × 10), in der Praxis nur einmal — danach Cache.

### Block: Zelle 5 — `compute_faithfulness`
```python
def compute_faithfulness(extraction, gt_contributions):
    gt_rank  = {c['feature']: i for i, c in enumerate(gt_contributions)}
    gt_sign  = {c['feature']: (1 if c['contribution']>=0 else -1) for c in gt_contributions}
    gt_value = {c['feature']: c['value'] for c in gt_contributions}

    for feat, info in extraction.items():
        if feat_key not in gt_rank: continue  # nicht in Top-K → überspringen
        # RA: rank match
        # SA: sign match
        # VA: value match (mit Toleranz, beidseitige Denormalisierung)
    return {'RA':..., 'SA':..., 'VA':...}
```
- **[Originalfunktion]** Implementiert Gleichung 1 aus dem Paper.
- **[Wichtige Designentscheidung]** Iteration über `extraction.items()` (was das LLM erwähnt hat) — **nicht** über die GT-Top-K (was es hätte erwähnen sollen).
- **[Selection-Bias]** Im Notebook in Abschnitt 4.1 explizit dokumentiert: Wer nur 1 Feature erwähnt und Sign trifft → SA = 1.0. Misst **Präzision, nicht Recall**. Paper-konform, aber JSON→Text wird bevorteilt, weil es lange Listen erwähnt.
- **[Bug-Verdacht / Halbe Information]** `gt_value` wird mit dem Wert befüllt, der in `feature_values` steht (also normalisiert für `temp`/`hum`/`windspeed`). `is_value_match` prüft **beide** Skalen — das fängt die häufige LLM-Praxis ab, Temperaturen in °C zu zitieren.

### Block: Zelle 6 — Detail-Tabelle
- Absolute Trefferquoten (z. B. `RA: 18/35`).

### Block: Zelle 7 — Visualisierung
- 3-Spalten-Subplot RA/SA/VA pro Pipeline.

### Block: Zelle 8 — Assumptions extrahieren
- Schreibt CSV mit allen extrahierten Annahmen pro (Pipeline, Modell, Instanz, Feature).
- Druckt Beispiele.

### Block: Zelle 9 — Perplexitäts-Berechnung (optional)
```python
PPL_MODEL_ID = 'mistralai/Mistral-7B-v0.3'
_tok = AutoTokenizer.from_pretrained(PPL_MODEL_ID)
_mdl = AutoModelForCausalLM.from_pretrained(PPL_MODEL_ID, device_map=device, load_in_4bit=True)
def compute_perplexity(text):
    inputs = _tok(text, return_tensors='pt').to(device)
    loss = _mdl(**inputs, labels=inputs['input_ids']).loss
    return float(torch.exp(loss).item())
```
- **[Originalfunktion]** Misst, wie „natürlich" eine Annahme klingt (niedrige PPL = plausibel).
- **[Voraussetzungen]** `transformers`, `accelerate`, `bitsandbytes` + GPU/MPS.
- **[Risiken]** `load_in_4bit=True` erfordert `bitsandbytes` — auf macOS (MPS) **nicht** unterstützt → Zelle würde dort fehlschlagen. Der Code prüft zwar `mps/cuda/cpu`, aber **nicht** die `bitsandbytes`-Verfügbarkeit. Praktisch im Repo wahrscheinlich nie ausgeführt (kein Output im `results/`-Ordner zu PPL).

### Block: Zelle 10 — Human Similarity (Proxy)
```python
emb_model = SentenceTransformer('all-MiniLM-L6-v2')
# Cross-Pipeline cosine-similarity:
for xai in XAI_MODELS:
    for iid in INSTANCE_IDS:
        texts = [explanation_04, explanation_05, explanation_06]
        embs = emb_model.encode(texts)
        sims = cosine_similarity(embs)
        # Mittelwert der off-diagonalen Einträge
```
- **[Originalfunktion]** Da keine menschlichen Referenznarrative existieren, wird Cross-Pipeline-Konsistenz gemessen.
- **[Semantik]** Hoher Wert = die drei Pipelines erklären dieselbe Instanz ähnlich. Schwacher Proxy für „die Pipelines konvergieren auf eine wahre Erklärung".
- **[Caveat im Kommentar]** Hat **kein** direktes Äquivalent im Paper — nur informativ.

### Block: Zelle 11 — Zusammenfassung + Statusliste
- Print mit `✓` / `⚠` / `✗` für die drei Paper-Metriken (Faithfulness implementiert; Perplexität bedingt; Human-Similarity ✗ ohne Referenznarrative).

### TL;DR Notebook 08
Setzt das Paper-Framework korrekt um (einschließlich des dokumentierten Selection-Bias) plus zwei Annäherungen für die nicht-direkt-verfügbaren Teile. Wissenschaftlich solide.

---

# 5. Modul-Zusammenfassungen

## Modul `utils/`

**Verantwortlichkeit:** Single-Source-of-Truth für Pfade, Daten-Vertrag, Modell-Registry, Erklärungs-Builder, LLM-IO, Tool-Box.

**Datenfluss (intern):**
```
__init__.py
   ├── data.py        ── load_train_test()
   ├── models.py      ── LOSS_OPTIONS, compute_metrics, save/load
   ├── explanations.py── FEATURE_SCHEMA, build_global, build_local (SHAP-Cache)
   ├── llm.py         ── ask_text, ask_with_images (Retry, Prompt-Cache)
   └── tools.py       ── TOOL_DEFINITIONS, ToolBox (dispatch über Reflection)
```

**Designentscheidungen:**
1. Dtypes ausschließlich in `data.py` rekonstruieren → keine Drift zwischen Notebooks.
2. `@dataclass(frozen=True)` für `LossOption` → immutable Konfiguration.
3. Module-Level SHAP-Cache → vermeidet teures Re-Initialize.
4. Tool-Use-Schleife bewusst im Notebook 06 statt in `llm.py` → projektspezifisch.
5. Reflection-Dispatching in `ToolBox` → einfache Erweiterbarkeit.

## Modul `notebooks/`

**Verantwortlichkeit:** Pipeline-Orchestrierung von Rohdaten bis Evaluation.

**Datenfluss:**
```
01 → data/{train,test}.csv
02a → models/*.pkl + results/model_metrics_*.json
02b → results/model_comparison_summary.csv
03 → explanations/*.json
04 → results/pipeline04/*.json
05 → explanations/plots/*.png + results/pipeline05/*.json
06 → results/pipeline06/*.json
07 → results/eval_*.{csv,png,json}
08 → results/eval08_ichmoukhamedov/*
```

**Zentrale Abhängigkeiten:**
- pandas (Daten), shap (XGB-Erklärungen), interpret (EBM), anthropic (LLM), joblib (Persistenz), sentence-transformers (optional 08).

**Designentscheidungen:**
1. **Notebook-getriebener Workflow** ohne CLI → für Forschung pragmatisch.
2. **`category`-Dtypes überall via `load_train_test`** → keine Notebook-lokalen dtype-Spielereien.
3. **Drei klar getrennte Pipelines** (04/05/06) statt einer parametrisierten → Vergleich ist trivial darstellbar.
4. **Idempotenz durch Caching** in 07 (`eval_llm_judge.json`) und 08 (`extractions.json`) → wiederholbare Läufe ohne API-Kosten.
5. **Multi-Judge-Strategie** in 07 (v1 Sonnet unkalibriert / v2 Sonnet kalibriert / v3 Opus unabhängig) → adressiert Self-Preference- und Rubrik-Bias.
6. **Random-Split** trotz Zeitreihendaten → bewusste Wahl mit dokumentierter Einschränkung.

## Technische Schulden (priorisiert)

| # | Punkt | Datei:Zeile | Aufwand |
|---|---|---|---|
| 1 | `_RETRYABLE`-Substring-Match → fragil; auf `isinstance(anthropic.APIError)` umstellen | [utils/llm.py:32,43](utils/llm.py:32) | klein |
| 2 | Legacy `save_ebm/load_ebm/save_xgb/load_xgb` entfernen (Dateien existieren nicht) | [utils/models.py:131-173](utils/models.py:131) | klein |
| 3 | `_tool_get_prediction`-Description nennt `season`/`workingday` (gibt es nicht mehr) | [utils/tools.py:60-65](utils/tools.py:60) | trivial |
| 4 | Toter Code `y_train = self.X_train.copy()` | [utils/tools.py:416](utils/tools.py:416) | trivial |
| 5 | `_shap_cache` mit `id(model)` → bei GC unsicher; `WeakKeyDictionary` | [utils/explanations.py:96-104](utils/explanations.py:96) | klein |
| 6 | `PROMPTS_DIR` zeigt auf nicht-existenten Ordner | [utils/__init__.py:18](utils/__init__.py:18) | trivial |
| 7 | `contribution_space="log"` hartkodiert → aus `LossOption.contribution_space` ableiten | [utils/explanations.py:277](utils/explanations.py:277) | mittel |
| 8 | `_encode_image` ohne Größenprüfung | [utils/llm.py:117-138](utils/llm.py:117) | klein |
| 9 | KNN-Encoding für nominale Features via `.cat.codes` semantisch fragwürdig | [utils/tools.py:392-401](utils/tools.py:392) | mittel |
| 10 | 02a trainiert Option 2 und 3 doppelt (identische Modelle) | [02a_Modeling_AllOptions.ipynb](notebooks/02a_Modeling_AllOptions.ipynb) Zelle 5 | mittel |
| 11 | NB 05 `list(X_test.index).index(iid)` fragil — funktioniert nur weil Index 0..N-1 ist | [05_LLM_Vision_Pipeline.ipynb](notebooks/05_LLM_Vision_Pipeline.ipynb) Zelle 3 | klein |
| 12 | NB 08 Zelle 9: `load_in_4bit=True` schlägt auf MPS fehl → Vorab-Check fehlt | [08_Evaluation_Ichmoukhamedov.ipynb](notebooks/08_Evaluation_Ichmoukhamedov.ipynb) Zelle 9 | klein |
| 13 | Keine Unit-Tests | repo-weit | groß |
| 14 | Kommentar in 02b „Notebook 02abc" referenziert altes Layout | [02b_Comparison.ipynb](notebooks/02b_Comparison.ipynb) Zelle 1 | trivial |

---

# 6. Gesamtfazit

## Gesamtarchitektur in Klartext

Drei-Schicht-Architektur — `utils/` (Domain-Logik) → `notebooks/` (Pipeline-Orchestrierung) → `data/`, `models/`, `explanations/`, `results/` (Artefakte). Keine zirkulären Imports, sauberer Single-Source-of-Truth-Ansatz. Alle drei LLM-Pipelines sind bewusst **vergleichbar** gehalten (gleiche Test-Instanzen, gleicher Loss, gleiches Sprachregister, gleiche Output-Struktur), Unterschiede liegen ausschließlich in der LLM-Eingabe-Modalität.

## Reihenfolge des Laufzeitflusses

1. `01_Data_Preprocessing.ipynb` — `hour.csv` → `train.csv`/`test.csv` (entfernt Leakage, Redundanz, hohe Korrelation; setzt category-Dtypes).
2. `02a_Modeling_AllOptions.ipynb` — trainiert 6 Modelle (XGB/EBM × 3 Losses), speichert je `.pkl` und `model_metrics_*.json`.
3. `02b_Comparison.ipynb` — konsolidiert Metriken, validiert Option-2/3-Identität, exportiert `model_comparison_summary.csv`.
4. `03_Explanations_Generation.ipynb` — SHAP für XGB, EBM-Terme für EBM; je 1 global + 10 local pro Modell = 22 JSONs.
5. `04_LLM_JSON_Pipeline.ipynb` — 20 LLM-Calls mit gecachtem System-Prompt; Output: `results/pipeline04/*.json`.
6. `05_LLM_Vision_Pipeline.ipynb` — 20 Waterfall-PNGs erzeugen, dann 20 multimodale LLM-Calls.
7. `06_LLM_ToolUse_Pipeline.ipynb` — 20 agentic loops, durchschnittlich 5,85 Tool-Calls; Output: `results/pipeline06/*.json`.
8. `07_Evaluation.ipynb` — quantitativer Vergleich, Keyword-Faithfulness, drei LLM-as-Judge-Runden (v1/v2/Opus).
9. `08_Evaluation_Ichmoukhamedov.ipynb` — formale RA/SA/VA-Metriken + Assumptions-Extraktion.

## Wichtigster End-to-End-Use-Case

Test-Instanz `inst=2058` („Fr, Okt, 05h, ~14°C, 2011", echte Ausleihe 31) wird:
1. mit `xgb_poisson_log.pkl` prognostiziert (Vorhersage z. B. ≈ 25).
2. SHAP-Werte werden im Log-Raum berechnet, oben dominiert `hr=5` mit großem negativen Beitrag.
3. Drei Pipelines erzeugen drei deutsche Erklärungen → 04 sieht JSON, 05 sieht Plot, 06 ruft Tools.
4. Notebook 07 bewertet die drei Erklärungen mit drei Judges; Notebook 08 misst RA/SA/VA.
5. Ergebnis in den Aggregaten: alle drei Pipelines erreichen Completeness ≈ 5; Faithfulness divergiert je nach Judge-Strenge.

## Liste kritischer Dateien

1. [utils/data.py](utils/data.py) — dtype-Vertrag (Bruch hier bricht alles)
2. [utils/explanations.py](utils/explanations.py) — Schema + Builder, die LLM und Tools sehen
3. [utils/tools.py](utils/tools.py) — Tool-Use-Backbone
4. [01_Data_Preprocessing.ipynb](notebooks/01_Data_Preprocessing.ipynb) — definiert Test-Set und damit `INSTANCE_IDS`-Bedeutung
5. [02a_Modeling_AllOptions.ipynb](notebooks/02a_Modeling_AllOptions.ipynb) — erzeugt die 6 `.pkl`-Files
6. [06_LLM_ToolUse_Pipeline.ipynb](notebooks/06_LLM_ToolUse_Pipeline.ipynb) — komplexester Pipeline-Notebook
7. [07_Evaluation.ipynb](notebooks/07_Evaluation.ipynb) — wissenschaftlicher Kern

## Liste typischer Fehlerquellen

1. **dtype-Drift**: CSV direkt lesen statt `load_train_test` → category-Verlust → XGBoost/EBM-Crash.
2. **iloc vs. Index-Verwechslung**: `INSTANCE_IDS` sind iloc; nach `reset_index` momentan identisch mit pandas-Index, aber bei Refactor gefährlich.
3. **Prompt-Cache-Schwelle**: < 1024 Tokens System-Prompt → kein Cache-Vorteil; in 04/05 bewusst über Feature-Schema-Inflation gelöst.
4. **API-Key fehlt**: `_get_client` mit gutem Hinweis, aber `.env` muss korrekt sein.
5. **Stale Modell-Files**: Legacy-Loader würden auf `ebm.pkl`/`xgb.pkl` zugreifen (existieren nicht).
6. **Counterfactual mit unbekannter Kategorie**: stilles NaN.
7. **Tool-Use `max_rounds` erreicht**: `last_text`-Fallback maskiert das.
8. **Judge sieht keine Tool-Outputs** (Opus-v3): strukturell unfair gegen Pipeline 06.
9. **NB 05 `list(...).index(iid)`**: O(n), bricht wenn Index nicht 0..N-1.
10. **NB 08 4-bit Mistral**: schlägt auf macOS (MPS) ohne CUDA fehl.

## Verbesserungsvorschläge (priorisiert)

| Prio | Maßnahme |
|---|---|
| **Hoch** | Unit-Tests für `data._apply_dtypes`, `models.compute_metrics`, `tools.ToolBox._build_input_df` (Punkt 13). |
| **Hoch** | 02a refactorn: Option 3 sollte Modell von Option 2 wiederverwenden, keine Doppeltrainings (Punkt 10). |
| **Mittel** | `contribution_space` aus `LossOption` ableiten (Punkt 7) — macht künftige Squared-Error-Pipelines korrekt. |
| **Mittel** | `_RETRYABLE` auf `isinstance(anthropic.APIError)` umbauen (Punkt 1). |
| **Mittel** | Notebook 08: vor 4-bit-Mistral-Load Plattform/Library-Verfügbarkeit prüfen (Punkt 12). |
| **Mittel** | Notebook 05: `pos = X_test.index.get_loc(iid)` statt `list().index()` (Punkt 11). |
| **Niedrig** | Tot-Code/Doku-Drift bereinigen: Punkte 2, 3, 4, 6, 14. |
| **Niedrig** | `_encode_image`: Größenwarnung bei > N MB (Punkt 8). |
| **Optional** | Fairere RA/SA/VA-Implementierung in NB 08, die über GT iteriert statt über Extraktion (im Notebook selbst als „nicht implementiert, da Paper-konform" markiert). |
| **Optional** | Faithfulness-Keywords in 07 als externe Konstante / Datei → einfacher pflegbar. |

## Wissenschaftliche Qualität

Die Implementierung ist **forschungsmäßig vorbildlich**:
- explizite Begründung jeder Methodenentscheidung (z. B. Random-Split, Loss-Wahl, Judge-Versionen),
- saubere Trennung von Modell-Ebene (02b) und Erklärungs-Ebene (07/08),
- dokumentierte methodische Einschränkungen in 07.17 und 08.4.1 (Self-Preference-Bias, Ceiling-Effekt, n=10, Selection-Bias der IM-Metriken),
- Multi-Judge-Strategie zur Bias-Kontrolle,
- ich-moukhamedov-konforme Implementierung trotz Adaptation an Regression statt Klassifikation.

Die Hauptverbesserungen liegen im **Code-Hygiene-Bereich** (Tests, Tot-Code, Doku-Drift), nicht in der wissenschaftlichen Methodik.
