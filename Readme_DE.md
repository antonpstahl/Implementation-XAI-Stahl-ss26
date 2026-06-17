# Implementierungszusammenfassung: LLM-gestützte XAI-Erklärungen für Fahrradverleih-Prognosen

## Überblick

Die Implementierung untersucht, wie Large Language Models (LLMs) genutzt werden können, um
Vorhersagen von Machine-Learning-Modellen automatisch in natürlichsprachliche Erklärungen
für Nicht-Experten zu übersetzen. Als Anwendungsfall dient das **Capital Bikeshare System**
in Washington D.C. — ein stündlicher Fahrradverleih-Datensatz.

Es werden **drei XAI-Pipelines** verglichen, die sich in der Art unterscheiden, wie das LLM
Erklärungsinformationen erhält: als strukturiertes JSON, als Bild (Waterfall-Plot) oder
über aktive Tool-Aufrufe.

---

## Projektstruktur

```
Implementation_1205/
├── data/               # Rohdaten und aufbereitete Train/Test-Splits
├── models/             # Trainierte Modelle (6 .pkl-Dateien)
├── explanations/       # SHAP-/EBM-Erklärungen als JSON + Waterfall-Plots (PNG)
├── results/            # Pipeline-Ausgaben, Evaluierungsplots, CSV-Zusammenfassungen
├── notebooks/          # 10 Jupyter Notebooks (00 Baseline, 01–08)
└── utils/              # Python-Hilfmodule (data.py, models.py, explanations.py, llm.py, tools.py)
```

---

## Schritt 1 — Datenaufbereitung (`01_Data_Preprocessing.ipynb`)

**Datensatz:** UCI Bike Sharing Dataset (Capital Bikeshare, Washington D.C.)
- 17 379 stündliche Beobachtungen, Januar 2011 bis Dezember 2012
- Zielgröße `cnt`: Anzahl der ausgeliehenen Fahrräder pro Stunde (1–977, Mittelwert ≈ 189)

**Verarbeitungsschritte:**

| Schritt | Aktion | Begründung |
|---|---|---|
| Leakage-Entfernung | `casual`, `registered`, `instant` entfernt | Direkte Teilsummen von `cnt` |
| Redundanz-Reduktion | `season` entfernt | Vollständig aus `mnth` ableitbar |
| Redundanz-Reduktion | `workingday` entfernt | Vollständig aus `weekday` + `holiday` ableitbar |
| Multikollinearität | `atemp` entfernt | Korrelation mit `temp` r ≈ 0,99 |
| Dtype-Kodierung | `mnth`, `hr`, `weekday`, `weathersit` → `category` | EBM und XGBoost nutzen native Kategorie-Splits |
| Zieltransformation | `cnt_log1p = log(1 + cnt)` | Skewness-Reduktion (2,44 → 0,17) |
| Split | 70 % Train / 30 % Test, zufällig | Gleichmäßige Jahresverteilung (2011/2012) |

**Verbleibende 9 Features:**

| Feature | Typ | Beschreibung |
|---|---|---|
| `hr` | ordinal | Stunde des Tages (0–23) |
| `mnth` | ordinal | Monat (1–12) |
| `weekday` | ordinal | Wochentag (0=Sonntag, 6=Samstag) |
| `weathersit` | nominal | Wetterlage (1=klar bis 4=Starkregen) |
| `yr` | binär | Jahr (0=2011, 1=2012) |
| `holiday` | binär | Feiertag (0/1) |
| `temp` | numerisch | Normalisierte Temperatur (÷41 → °C) |
| `hum` | numerisch | Normalisierte Luftfeuchtigkeit (÷100 → %) |
| `windspeed` | numerisch | Normalisierte Windgeschwindigkeit (÷67 → km/h) |

---

## Schritt 2 — Modellierung (`02a_Modeling_AllOptions.ipynb`, `02b_Comparison.ipynb`)

Zwei Modellklassen wurden jeweils mit drei Verlustfunktionen trainiert:

**Modelle:**
- **XGBoost** (`xgboost.XGBRegressor`, `enable_categorical=True`)
- **EBM** (Explainable Boosting Machine, `interpret.glassbox.ExplainableBoostingRegressor`)

**Verlustfunktionen (drei Optionen):**

| Option | Verlust | Besonderheit |
|---|---|---|
| 1 — Squared Error | `reg:squarederror` / `rmse` | Einfach; kann negative Vorhersagen liefern |
| 2 — Poisson-Log | `count:poisson` / `poisson_deviance` | Beiträge im Log-Raum; strikt positive Vorhersagen |
| 3 — Poisson-Native | Gleiches Modell wie Option 2 | Beiträge approximativ auf Ausleihe-Skala |

**Metriken auf dem Testset:**

<!-- AUTO-TABLE:model-comparison-de -->
| Option | Modell | RMSE | MAE | R² | Poisson-Dev. | Neg. Vorhersagen |
|---|---|---|---|---|---|---|
| Squared Error | XGB | 46,43 | 28,72 | 0,932 | 17,10 | 133 |
| Squared Error | EBM | 59,64 | 39,69 | 0,889 | 79,63 | 411 |
| **Poisson-Log** | **XGB** | **45,44** | **27,00** | **0,935** | **9,38** | **0** |
| **Poisson-Log** | **EBM** | **48,20** | **28,20** | **0,927** | **10,81** | **0** |
<!-- /AUTO-TABLE:model-comparison-de -->

(Testset n = 5 227; Werte aus `results/model_comparison_summary.csv`.)

**Gewählte Option für alle weiteren Schritte:** Poisson-Log (Option 2) — beste Poisson-Deviance,
keine negativen Vorhersagen, physikalisch korrekte Modellierung von Zähldaten.

---

## Schritt 3 — Erklärungsgenerierung (`03_Explanations_Generation.ipynb`)

Für beide Modelle (XGB, EBM) wurden globale und lokale Erklärungen erstellt:

**Globale Erklärungen** (gespeichert in `explanations/global_*.json`):
- XGBoost: SHAP-basierte Feature Importance (mean |SHAP|) über Trainingsset
- EBM: Term Importances aus den gelernten Funktionen (ohne Interaktionsterme)
- Top-Features beider Modelle: `hr` → `yr` / `temp` → `temp` / `weekday`

**Lokale Erklärungen** (10 Test-Instanzen, `explanations/local_*.json`):
- Stratifiziert über 5 cnt-Quintile (Bereich: 31–557 Ausleihen)
- SHAP-Werte (XGB) und EBM-Term-Beiträge im Log-Raum
- Waterfall-Plots als PNG (`explanations/plots/waterfall_*.png`)

**Test-Instanzen:**

| ID | cnt | Kontext |
|---|---|---|
| 224 | 270 | Do, Feb, 13h, klar, ~8°C, 2011 |
| 580 | 5 | So, Mär, 00h, klar, ~9°C, 2011 |
| 1041 | 229 | So, Mai, 10h, bewölkt, ~27°C, 2011 |
| 1481 | 113 | Sa, Jul, 08h, klar, ~32°C, 2011 |
| 1677 | 145 | Fr, Aug, 18h, bewölkt, ~30°C, 2011 |
| 2058 | 238 | Fr, Okt, 05h, klar, ~14°C, 2011 |
| 2510 | 337 | Mi, Dez, 10h, bewölkt, ~20°C, 2011 |
| 3543 | 691 | So, Mai, 09h, bewölkt, ~21°C, 2012 |
| 3847 | 122 | So, Jun, 20h, klar, ~25°C, 2012 |
| 4454 | 311 | Mi, Sep, 07h, klar, ~21°C, 2012 |

---

## Schritt 4 — Drei LLM-Pipelines + deterministische Baseline

Alle LLM-Pipelines verwenden `claude-sonnet-4-6` und erzeugen deutsche, dreistufige Erklärungen
(Abschnitte `[VORHERSAGE]`, `[TREIBER]`, `[EMPFEHLUNG]`) für Mitarbeitende ohne technischen Hintergrund.

### Pipeline 00 — Template-Baseline (`00_Template_Pipeline_Baseline.ipynb`)

Deterministischer Textbaustein-Generator, der dieselbe Dreiteilung aus denselben SHAP-/EBM-JSONs
ohne LLM-Aufruf befüllt. Beantwortet die Standard-Reviewer-Frage: *Was leistet das LLM über
einen Textbaustein hinaus?*

### Pipeline 04 — JSON → Text (`04_LLM_JSON_Pipeline.ipynb`)

Das LLM erhält globale Feature Importance und lokale SHAP-/EBM-Beiträge als strukturiertes JSON.

- **Eingabe:** JSON-Payload mit Metriken, Top-Features, Feature-Werten und Top-6-Beiträgen
- **System-Prompt:** Gecacht via Anthropic Prompt Caching (> 1 024 Tokens; enthält Domain-Kontext
  und Feature-Schema)
- **Besonderheit:** `build_context_string()` denormalisiert Rohwerte in Alltagssprache
  (z.B. `temp=0.68` → `~27,9 °C`) vor dem API-Aufruf

### Pipeline 05 — Vision → Text (`05_LLM_Vision_Pipeline.ipynb`)

Das LLM erhält den Waterfall-Plot der Instanz als base64-kodiertes PNG.

- **Eingabe:** Bild + kurzer Textprompt mit Instanz-ID, Vorhersage und tatsächlichem Wert
- **Methode:** `ask_with_images()` aus `utils/llm.py`; multimodale Anthropic API
- **Besonderheit:** Kein numerischer Zugriff auf Beitragswerte — das Modell liest Balkenlängen
  visuell ab (potenzielle Unschärfe bei kleinen Beiträgen)

### Pipeline 06 — Tool-Use (`06_LLM_ToolUse_Pipeline.ipynb`)

Das LLM ruft Daten selbst über definierte Tools ab (agentic loop).

- **Tools** (8 Funktionen, definiert in `utils/tools.py`):

| Tool | Funktion |
|---|---|
| `get_feature_schema` | Feature-Metadaten und Beschreibungen |
| `get_feature_importance` | Globale Importance (SHAP / EBM-Terme) |
| `get_prediction` | Vorhersage für beliebige Feature-Kombination |
| `get_shap_values` | Lokale Beiträge einer Test-Instanz |
| `get_partial_dependence` | PD-Kurve für ein Feature |
| `get_feature_value_context` | Perzentil und Statistiken eines Feature-Werts |
| `get_similar_instances` | K nächste Nachbarn (euklidisch, Min-Max-normiert) |
| `get_counterfactual_prediction` | Was-wäre-wenn-Vorhersage bei geänderten Features |

- **Ablauf:** Agentic loop bis `stop_reason == "end_turn"`; durchschnittlich **5,65 Tool-Calls**
  pro Erklärung

---

## Schritt 5 — Evaluation (`07_Evaluation.ipynb`, `08_Evaluation_Ichmoukhamedov.ipynb`)

### Quantitativer Vergleich

Mittelwerte über 20 Erklärungen pro Pipeline (2 XAI-Modelle × 10 Instanzen):

<!-- AUTO-TABLE:pipeline-quant-de -->
| Pipeline | Ø Wörter | Ø Input-Tokens¹ | Ø Output-Tokens | Gesamtkosten (20 Calls) | Ø Latenz |
|---|---|---|---|---|---|
| Template | 54 | 0 | 0 | 0,00 USD | 0,0 s |
| JSON→Text | 208 | 616 | 510 | 0,16 USD | 11,7 s |
| Vision | 212 | 2 167 | 528 | 0,29 USD | 12,3 s |
| Tool-Use | 305 | 3 489 | 1 225 | 0,58 USD | 28,8 s |
<!-- /AUTO-TABLE:pipeline-quant-de -->

¹ *Input-Tokens sind die abgerechneten, nicht gecachten Tokens. JSON→Text cacht den System-Prompt
(Cache-Read-Tokens, ~10 % Preis, hier nicht gezählt) — daher liegt der Wert weit unter den frisch
übertragenen Bild-Tokens von Vision.* Werte aus `results/eval_summary.csv`.

> **Hinweis zur Keyword-Faithfulness:** Die ursprüngliche keyword-basierte Faithfulness-Metrik
> wurde entfernt — sie lag am Ceiling (0,94–1,0) und differenzierte nicht zwischen Pipelines.
> Die Faithfulness-Bewertung stützt sich nun auf LLM-as-Judge und die formalen RA/SA/VA-Metriken.

### LLM-as-Judge (drei Judge-Versionen)

**v1-Scores (Sonnet, unkalibriert) 1–5 pro Kriterium, aus `results/eval_summary.csv`:**

<!-- AUTO-TABLE:judge-scores-de -->
| Pipeline | Faithfulness | Clarity | Completeness |
|---|---|---|---|
| Template | 5,00 | 4,70 | 4,00 |
| JSON→Text | 4,35 | 4,90 | 4,95 |
| Vision | 3,80 | 4,55 | 4,75 |
| Tool-Use | 4,40 | 3,95 | 4,90 |
<!-- /AUTO-TABLE:judge-scores-de -->

**Judge-Versionen:**
- **v1** (Sonnet, unkalibriert): Ceiling-Effekt ~91 % der Scores = 5; zu mildes Urteil
- **v2** (Sonnet, kalibrierte Rubrik): ~73 % Scores = 5; strukturiertere Differenzierung.
  Hinweis: v2 wurde auf einem anderen (Convenience-)Sample [42, 100, 250, 500, 1337] und **ohne
  Template** erhoben, daher nicht 1:1 mit v1/v3 vergleichbar.
- **v3** (Opus 4.8, unabhängiges Modell): strengstes Urteil. Ein systematischer Offset Opus < Sonnet
  ist *konsistent mit* einem Self-Preference-Bias, aber **nicht** dessen Beweis — beide Judges sind
  Anthropic-Modelle; ein echter Cross-Vendor-Judge steht noch aus (Implementierungsplan, Phase 2).

**Tool-Use-Kontext für den Judge:** v3 (Opus) erhält das vollständige Tool-Call-Transkript
(Aufrufe + Ergebnisse) als Teil von `ground_truth`, sodass per Tool abgerufene Zahlen
(PD-Kurven, Kontrafaktika, Perzentile) verifizierbar sind (Fix aus Plan-Phase 0).

### Ichmoukhamedov-Faithfulness (`08_Evaluation_Ichmoukhamedov.ipynb`)

Formale Faithfulness-Metriken nach Ichmoukhamedov et al. (2024), n = 10 Instanzen
(Präzisions-artige Metriken — Selection-Bias siehe NB 08 §4.1):

<!-- AUTO-TABLE:faithfulness-de -->
| Pipeline | RA (Rank) | SA (Sign) | VA (Value) |
|---|---|---|---|
| JSON→Text | 0,562 | 0,721 | 0,667 |
| Tool-Use | 0,558 | 0,733 | 0,733 |
| Vision | 0,429 | 0,679 | 0,575 |
<!-- /AUTO-TABLE:faithfulness-de -->

---

## Technische Details

### Abhängigkeiten

```
anthropic           # LLM-API-Client
xgboost             # Gradient Boosting
interpret           # EBM (InterpretML)
shap                # SHAP-Werte für XGBoost
scikit-learn        # Train/Test-Split
pandas, numpy       # Datenverarbeitung
matplotlib, seaborn # Visualisierungen
joblib              # Modell-Serialisierung
python-dotenv       # API-Key-Verwaltung
```

### Konfiguration

API-Key wird aus `.env` geladen (`ANTHROPIC_API_KEY=sk-ant-...`).
Alle Pfade sind relativ zur Projekt-Wurzel in `utils/__init__.py` definiert.
Reproduzierbarkeit durch `RANDOM_STATE = 42`.

### Testsuite

```bash
pytest tests/        # gesamte Testsuite
pytest tests/test_prompt_golden.py -v   # nur Prompt-Regression
```

Die Suite umfasst Sampling-Determinismus, Generierungs-Loop-Persistenz/Resume, Judge-Parsing-Robustheit, Statistikfunktionen, Denormalisierungs-Konsistenz, README-Konsistenz und **Prompt-Fix-Regression**.
Der Prompt-Regressionstest (`test_prompt_golden.py`) friert die SHA-256-Hashes und kritischen Constraint-Phrasen aller drei Pipeline-Prompts ein (Vorzeichen- und Rangtreue für `yr=0`, Phase-3-Fix).
Er ist ein hartes Gate: Phase 3b (Vollauf) darf erst starten, wenn alle Tests grün sind.

**Grünes-Gate-Status (Phase 3a):** `pytest tests/` → **139 passed** (2026-06-17, Python 3.13). Alle skalierungskritischen Pfade sind abgedeckt; Phase 3b ist damit freigegeben.

**Wenn ein Prompt absichtlich verbessert wird:**
1. Prompt-Datei bearbeiten.
2. Neuen Hash berechnen: `shasum -a 256 prompts/<datei>.md`
3. `GOLDEN_HASHES` in `tests/test_prompt_golden.py` aktualisieren.
4. Falls sich eine Constraint-Phrase geändert hat, auch `REQUIRED_PHRASES` anpassen.
5. `pytest tests/test_prompt_golden.py` muss grün sein.

### Ausführungsreihenfolge

```
01_Data_Preprocessing      → data/train.csv, data/test.csv
02a_Modeling_AllOptions    → models/*.pkl
02b_Comparison             → results/model_comparison_summary.csv
03_Explanations_Generation → explanations/*.json, explanations/plots/*.png
04_LLM_JSON_Pipeline       → results/pipeline04/*.json
05_LLM_Vision_Pipeline     → results/pipeline05/*.json
06_LLM_ToolUse_Pipeline    → results/pipeline06/*.json
07_Evaluation              → results/eval_*.{csv,png,json}
08_Evaluation_Ichmoukhamedov → results/eval08_ichmoukhamedov/
```

---

## Kernbefunde

> **Status dieser Befunde:** deskriptiv/explorativ. Bei n = 10–20 Erklärungen pro Pipeline,
> ohne Repeated Sampling und ohne Inferenzstatistik sind die folgenden Unterschiede **nicht**
> statistisch abgesichert (siehe Limitationen-Tabelle in `07_Evaluation.ipynb` §7). Richtungsweisend,
> nicht beweisend.

1. **Die deterministische Template-Baseline gewinnt bei Faithfulness** (5,00 vs. 3,80–4,40 der
   LLM-Pipelines): Sie nennt konstruktionsbedingt exakt die wahren Top-Treiber. Die LLMs tauschen
   etwas Treue gegen reichere, lesbarere Narrative ein — das ist der eigentliche Kernbefund zur
   Frage „Was leistet das LLM über einen Textbaustein hinaus?" und ein Trade-off, kein Gratis-Gewinn.

2. **Unter den LLM-Pipelines:** Faithfulness Tool-Use (4,40) ≈ JSON→Text (4,35) > Vision (3,80) —
   konsistent mit der formalen Rank-Agreement (Vision 0,43 vs. ~0,56). Visuelles Ablesen von
   Balkenlängen ist strukturell ungenauer als numerischer Zugriff.

3. **Clarity und Completeness liegen für die LLM-Pipelines am Ceiling** (≥ 4,55 / ≥ 4,75) und
   differenzieren nicht; die Template-Baseline fällt bei Completeness ab (4,00, dünnere Empfehlung).

4. **JSON→Text** ist am effizientesten (≈ 0,008 USD/Erklärung, niedrigste Latenz 11,7 s) —
   System-Prompt-Caching hält die abgerechneten Input-Tokens niedrig.

5. **Tool-Use** generiert die längsten Erklärungen (+47 % Wörter ggü. JSON→Text) mit quantitativen
   Belegen aus PD-Kurven und Kontrafaktika — zu ~3,6× Kosten und ~2,5× Latenz.

6. **Vision** liegt bei der Latenz nah an JSON→Text, kostet aber mehr (Bild-Tokens, kein
   Caching-Vorteil) und hat die niedrigste Faithfulness aller Pipelines.

7. **Möglicher Self-Preference-Bias:** Opus-Scores (v3) liegen systematisch unter Sonnet (v1/v2)
   bei identischer Rubrik. Konsistent mit einem Self-Preference-Effekt, aber nicht beweisend —
   beide Judges sind Anthropic-Modelle; ein Cross-Vendor-Judge steht aus.
