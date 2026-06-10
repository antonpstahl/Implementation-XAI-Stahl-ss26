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
├── notebooks/          # 8 Jupyter Notebooks (01–08)
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

| Option | Modell | RMSE | MAE | R² | Poisson-Dev. | Neg. Vorhersagen |
|---|---|---|---|---|---|---|
| Squared Error | XGB | 39,13 | 24,31 | 0,952 | 12,03 | 108 |
| Squared Error | EBM | 55,31 | 36,19 | 0,903 | 59,99 | 358 |
| **Poisson-Log** | **XGB** | **39,01** | **23,68** | **0,952** | **7,06** | **0** |
| **Poisson-Log** | **EBM** | **48,37** | **27,00** | **0,926** | **9,72** | **0** |

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

## Schritt 4 — Drei LLM-Pipelines

Alle Pipelines verwenden `claude-sonnet-4-6` und erzeugen deutsche, dreistufige Erklärungen
(Abschnitte `[VORHERSAGE]`, `[TREIBER]`, `[EMPFEHLUNG]`) für Mitarbeitende ohne technischen Hintergrund.

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

- **Ablauf:** Agentic loop bis `stop_reason == "end_turn"`; durchschnittlich **5,85 Tool-Calls**
  pro Erklärung

---

## Schritt 5 — Evaluation (`07_Evaluation.ipynb`, `08_Evaluation_Ichmoukhamedov.ipynb`)

### Quantitativer Vergleich

| Pipeline | Ø Wörter | Ø Input-Tokens | Ø Output-Tokens | Gesamtkosten (20 Calls) | Ø Latenz |
|---|---|---|---|---|---|
| JSON→Text | 211 | 1 837 | 511 | 0,26 USD | 12,2 s |
| Vision | 207 | 2 187 | 512 | 0,28 USD | 12,3 s |
| Tool-Use | 306 | 3 427 | 1 263 | 0,58 USD | 29,8 s |

### Faithfulness (Keyword-basiert)

Anteil der Top-3-Features, die in der Erklärung erwähnt werden:

| Pipeline | Ø Faithfulness |
|---|---|
| JSON→Text | **1,000** |
| Vision | **1,000** |
| Tool-Use | 0,984 |

### LLM-as-Judge (drei Judge-Versionen)

**Scores 1–5 pro Kriterium:**

| Pipeline | Faithfulness (v1) | Faithfulness (v2) | Faithfulness (Opus) | Clarity (v2) | Completeness (v2) |
|---|---|---|---|---|---|
| JSON→Text | 4,55 | 3,80 | 3,50 | 4,70 | **5,00** |
| Tool-Use | 4,62 | **4,60** | 3,70 | 4,90 | **5,00** |
| Vision | 4,24 | 4,40 | **4,40** | 4,70 | 4,90 |

**Judge-Versionen:**
- **v1** (Sonnet, unkalibriert): Ceiling-Effekt 91 % der Scores = 5; zu mildes Urteil
- **v2** (Sonnet, kalibrierte Rubrik): 73 % Scores = 5; strukturiertere Differenzierung
- **v3** (Opus 4.7, unabhängiges Modell): 50 % Scores = 5; strengstes Urteil;
  Self-Preference-Bias aus v1/v2 ausgeschlossen

**Methodische Einschränkung Opus für Tool-Use:** Der Judge sieht keine Tool-Call-Outputs
(PD-Kurven, Kontrafaktika); Zahlen, die das LLM korrekt per Tool abgerufen hat, gelten als
nicht belegbar → strukturell niedrigere Scores trotz korrekter Inhalte.

### Ichmoukhamedov-Faithfulness (`08_Evaluation_Ichmoukhamedov.ipynb`)

Formale Faithfulness-Metriken nach Ichmoukhamedov et al. (2021):
- **RA** (Rank Agreement): Übereinstimmung der Feature-Rangfolge zwischen Modell und Erklärung
- **SA** (Sign Agreement): Korrekte Wirkungsrichtung (positiv/negativ)
- **VA** (Value Agreement): Numerische Nähe der genannten Beitragswerte

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

1. **Completeness** ist robust: Alle drei Pipelines erzielen ≥ 4,9/5 — der dreistufige
   Aufbau (Vorhersage / Treiber / Empfehlung) wird zuverlässig eingehalten.

2. **Faithfulness** differenziert je nach Judge-Strenge: Vision ist konsistent bei ≈ 4,4,
   während JSON→Text und Tool-Use je nach Rubrik stark variieren.

3. **JSON→Text** ist am effizientesten (≈ 0,013 USD/Erklärung) und erreicht beim
   unkalibrieren v1-Judge den höchsten Gesamtscore (14,30/15).

4. **Tool-Use** generiert längere Erklärungen (+46 % Wörter) mit quantitativen
   Belegen aus PD-Kurven und Kontrafaktika — zu erheblich höheren Kosten (×2,2) und
   Latenz (×2,4).

5. **Vision** liegt bei Kosten und Latenz nah an JSON→Text, hat aber strukturell
   niedrigere Faithfulness-Potenziale, da Balkenlängen visuell und unscharf abgelesen werden.

6. **Self-Preference-Bias** im Judge ist messbar: Opus-Scores liegen systematisch unter
   Sonnet-Scores bei identischer Rubrik. Der Effekt ist bei Tool-Use am stärksten.
