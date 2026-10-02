# LLM-gestützte Erklärung von Machine-Learning-Modellausgaben: Der Einfluss von XAI-Methode und Übergabeformat auf die Erklärungsqualität

## Überblick

Die Implementierung untersucht, wie Large Language Models (LLMs) genutzt werden können, um
Vorhersagen von Machine-Learning-Modellen automatisch in natürlichsprachliche Erklärungen
für Nicht-Experten zu übersetzen. Als Anwendungsfall dient das **Capital Bikeshare System**
in Washington D.C., ein stündlicher Fahrradverleih-Datensatz.

Es werden **drei Übergabeformate** verglichen, die sich darin unterscheiden, wie das LLM
Erklärungsinformationen erhält: als strukturiertes JSON, als Bild (Plot) oder über aktive
Tool-Aufrufe, jeweils gegen eine deterministische Template-Baseline.

### Zwei Tracks

Nach dem Betreuungsgespräch vom 30.06. läuft das Projekt auf **zwei Tracks**, wobei der
**globale Track die Kernaussage trägt**:

| Track | Erklärungseinheit | Notebooks | n | Rolle |
| --- | --- | --- | --- | --- |
| **Global** (Haupttrack) | wie das Modell ein **Feature** nutzt, und das **Gesamtmodell** | `04Ga-04Ge`, `05G`, `05Gb` | 72 Per-Feature- + 90 Whole-Model-Records | trägt die Befunde |
| **Lokal** (Vergleichsbasis) | warum **eine Instanz** ihre Vorhersage bekam | `04La-04Ld`, `05`, `06` | 20 Erklärungen je Pipeline | Vergleichsbasis, reproduzierbar gehalten |

Der globale Track bewertet jede Beschreibung gegen eine **strukturierte Ground Truth**
(18 Referenzen, alle 18 mechanisch verifiziert) mit zwei unabhängigen Scorern: einem
deterministischen Keyword-**Rubric** und einem referenz-gestützten **LLM-Judge**, letzterer
über **zwei Vendor** (Anthropic Opus + OpenAI) als Robustheitscheck.

---

## Projektstruktur

```
llm-assisted-xai-explanations/
├── data/               # Rohdaten und aufbereitete Train/Test-Splits
├── models/             # Trainierte Modelle (6 .pkl-Dateien)
├── explanations/       # Lokale SHAP-/EBM-Erklärungen + globale Kurven, Beeswarms, Ground Truth
├── results/            # Pipeline-Ausgaben, Judge-Scores, Evaluierungsplots, CSV-Zusammenfassungen
├── notebooks/          # 18 Jupyter Notebooks (01-06; 04Ga-Gf global, 04La-Ld lokal)
├── analyses/           # GT-Verifikationsreport, Fehleranalyse nach Formtyp
├── planning/           # Revisionsplan, Korrekturenlog, Limitationen
├── prompts/            # Prompt-Vorlagen (lokal + global + Judge)
└── utils/              # Python-Hilfsmodule (data, models, explanations, llm, tools,
                        #   global_feature, global_whole, groundtruth, rubric, global_eval)
```

`results/` hält beide Tracks nebeneinander, damit jede in der Arbeit zitierte Zahl aus
**einem** Baum reproduzierbar ist:

```
results/
├── global/                    # G2a: 72 Per-Feature-Beschreibungen (4 Formen × 2 Modelle × 9 Features)
├── global_rubric.csv          # G2a: deterministische Rubric-Scores
├── global_judge{,_openai}/    # G2a: referenz-gestützter Judge, 2 Vendor
├── global_whole/              # G2b: 10 Whole-Model-Antworten (5 Bedingungen × 2 Modelle)
├── global_whole_split/        # G2b: dieselben Antworten, in 90 Per-Feature-Records zerlegt
├── global_whole_judge{,_openai}/   # G2b: Judge auf den Split-Records, 2 Vendor
├── pipeline00/04/05/06/       # lokaler Track: 20 Erklärungen je Pipeline
└── eval_*.{csv,json,png}, eval06_ichmoukhamedov/   # Evaluation des lokalen Tracks
```

---

## Schritt 1: Datenaufbereitung (`01_Data_Preprocessing.ipynb`)

**Datensatz:** UCI Bike Sharing Dataset (Capital Bikeshare, Washington D.C.)
- 17 379 stündliche Beobachtungen, Januar 2011 bis Dezember 2012
- Zielgröße `cnt`: Anzahl der ausgeliehenen Fahrräder pro Stunde (1-977, Mittelwert ≈ 189)

**Verarbeitungsschritte:**

| Schritt | Aktion | Begründung |
|---|---|---|
| Leakage-Entfernung | `casual`, `registered`, `instant` entfernt | Direkte Teilsummen von `cnt` |
| Redundanz-Reduktion | `season` entfernt | Vollständig aus `mnth` ableitbar |
| Redundanz-Reduktion | `workingday` entfernt | Vollständig aus `weekday` + `holiday` ableitbar |
| Multikollinearität | `atemp` entfernt | Korrelation mit `temp` r ≈ 0,99 |
| Dtype-Kodierung | `mnth`, `hr`, `weekday`, `weathersit` → `category` | EBM und XGBoost nutzen native Kategorie-Splits |
| Zielvariable | rohes `cnt` mit Poisson-Deviance-Loss (Log-Link) | Schiefe wird intern durch die Log-Link-Fitfunktion absorbiert, keine explizite `log1p`-Transformation nötig |
| Split | 70 % / 30 %, **auf Tagesebene** randomisiert (511 Train-, 220 Test-Tage von 731, keine Überschneidung) | Ganze Tage als Einheit → benachbarte Stunden desselben Tages landen nie in Train *und* Test; gleichmäßige Jahresverteilung (2011/2012) |

**Verbleibende 9 Features:**

| Feature | Typ | Beschreibung |
|---|---|---|
| `hr` | ordinal | Stunde des Tages (0-23) |
| `mnth` | ordinal | Monat (1-12) |
| `weekday` | ordinal | Wochentag (0=Sonntag, 6=Samstag) |
| `weathersit` | nominal | Wetterlage (1=klar bis 4=Starkregen) |
| `yr` | binär | Jahr (0=2011, 1=2012) |
| `holiday` | binär | Feiertag (0/1) |
| `temp` | numerisch | Normalisierte Temperatur (÷41 → °C) |
| `hum` | numerisch | Normalisierte Luftfeuchtigkeit (÷100 → %) |
| `windspeed` | numerisch | Normalisierte Windgeschwindigkeit (÷67 → km/h) |

---

## Schritt 2: Modellierung (`02a_Modeling_AllOptions.ipynb`, `02b_Comparison.ipynb`)

Zwei Modellklassen wurden jeweils mit drei Verlustfunktionen trainiert:

**Modelle:**
- **XGBoost** (`xgboost.XGBRegressor`, `enable_categorical=True`)
- **EBM** (Explainable Boosting Machine, `interpret.glassbox.ExplainableBoostingRegressor`)

**Verlustfunktionen (drei Optionen):**

| Option | Verlust | Besonderheit |
|---|---|---|
| 1. Squared Error | `reg:squarederror` / `rmse` | Einfach; kann negative Vorhersagen liefern |
| 2. Poisson-Log | `count:poisson` / `poisson_deviance` | Beiträge im Log-Raum; strikt positive Vorhersagen |
| 3. Poisson-Native | Gleiches Modell wie Option 2 | Beiträge approximativ auf Ausleihe-Skala |

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

**Gewählte Option für alle weiteren Schritte:** Poisson-Log (Option 2): beste Poisson-Deviance,
keine negativen Vorhersagen, physikalisch korrekte Modellierung von Zähldaten.

---

## Schritt 3: Erklärungsgenerierung (`03_Explanations_Generation.ipynb`)

> Dieses Notebook speist **beide** Tracks: lokale Instanz-Erklärungen (SHAP-/EBM-JSON +
> Waterfall-Plots) für den lokalen Track und die globalen Kurven-Artefakte
> (Shape/Dependence je Feature, Beeswarms) für den globalen Track.

Für beide Modelle (XGB, EBM) wurden globale und lokale Erklärungen erstellt:

**Globale Erklärungen** (gespeichert in `explanations/global_*.json`):
- XGBoost: SHAP-basierte Feature Importance (mean |SHAP|) über Trainingsset
- EBM: Term Importances aus den gelernten Funktionen (ohne Interaktionsterme)
- Top-Features beider Modelle: `hr` → `yr` / `temp` → `temp` / `weekday`

**Lokale Erklärungen** (10 Test-Instanzen, `explanations/local_*.json`):
- Stratifiziert über 5 cnt-Quintile (Bereich: 31-557 Ausleihen)
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

## Globaler Track (Haupttrack): Wie nutzt das Modell seine Features?

Die Erklärungseinheit ist hier nicht eine einzelne Vorhersage, sondern die **Nutzung eines
Features durch das Modell**: und in `04Ge` das **Gesamtmodell in einem Aufruf**. Jede
Beschreibung wird gegen eine strukturierte Ground Truth bewertet.

**Globale Artefakte (G0/G1, aus `03_Explanations_Generation.ipynb`).** Je Feature eine
Shape-/Dependence-Kurve und ein Beeswarm (`explanations/global_curve_*.json`,
`explanations/plots/global/`). Für den EBM, dessen Bibliothek keinen Beeswarm liefert, wird
einer aus den Shape-Funktionen **konstruiert** und gegen den XGB-SHAP-Swarm validiert
(Rang-Spearman 0,883, Richtungsübereinstimmung 100 %).

**Ground Truth (G3)**: `explanations/global_groundtruth/`, 18 Referenzen = 9 Features ×
2 XAI-Modelle. Je Feature: Formtyp (`monotonic` / `non-monotonic` / `categorical` /
`near-flat`), Richtung, Importance-Rang, Peak, Top-Kategorien. Mechanisch aus den
G0-Kurven abgeleitet über **einen geteilten** `classify_shape`-Helfer, denselben, den auch
die deterministische Baseline aufruft, sodass beide nicht mehr divergieren können. Alle 18
Referenzen bestehen eine mechanische Re-Ableitung (`analyses/gt_verification.md`); eine
Schwellen-Sensitivitätsanalyse lässt 15/18 Labels über das gesamte Gitter stabil, die drei
Grenzfälle sind namentlich ausgewiesen.

### G2a: ein Feature pro Aufruf (`04Ga`–`04Gd`)

Vier Übergabeformen × 2 XAI-Modelle × 9 Features = **72** Beschreibungen, bewertet vom
deterministischen Rubric **und** vom referenz-gestützten Judge (zwei Vendor):

<!-- AUTO-TABLE:global-feature-de -->
| Form | Rubric gesamt | Judge Faithfulness | Clarity | Completeness | Faithfulness (OpenAI) |
|---|---|---|---|---|---|
| Template | 0,944 | 4,50 | 4,06 | 5,00 | 4,72 |
| JSON | 0,889 | 4,28 | 4,00 | 5,00 | 3,44 |
| Vision | 0,907 | 4,61 | 3,67 | 5,00 | 3,83 |
| Tool Use | 0,880 | 4,44 | 3,67 | 5,00 | 3,83 |
<!-- /AUTO-TABLE:global-feature-de -->

Auf dem Anthropic-Judge führt `vision` mit 4,61 nominal, die Template-Baseline liegt
mit 4,50 auf Platz zwei; keiner der sechs Paarvergleiche ist signifikant (Wilcoxon
signed-rank über `(feature, xai_model)`-Paare, Holm-korrigiert, alle `p_adj ≥ 0,38`,
fünf von sechs Cliff's *d* „negligible", `json`–`vision` klein bei δ = −0,17). Auf
dem OpenAI-Judge liegt die Template-Baseline vor allen drei LLM-Formaten, und
template–`json` erreicht `p_adj = 0,030` bei δ = 0,57; die Überschreitung hängt
jedoch an einem einzigen Urteil (0,067, wenn `json_ebm_weathersit` mit 5 statt 3
bewertet wird) und übersteht die Korrektur über beide Judges nicht (0,061).

> **Framing-Auflage:** Seit der Schwellen-Vereinheitlichung trifft die Baseline den Formtyp
> *by construction*. Sie ist damit ein **strenger Referenz-Boden**, kein unabhängiger
> Konkurrent. Die LLMs bekommen die abgeleitete Klasse nie, sie lesen Rohkurve, Plot oder
> Tool-Ausgabe und müssen near-flat selbst erkennen; ihre Fehler dort sind echt.
> Und: Nicht-Signifikanz ist **keine** belegte Äquivalenz (dafür bräuchte es einen
> TOST-Äquivalenztest), sondern „kein nachweisbarer Unterschied".

Rubric und Judge korrelieren mit Spearman 0,458 (n = 72, p < 0,001), interne Konsistenz
zweier GT-gebundener Scorer, keine unabhängige Kriteriumsvalidität.

Der belastbare Befund ist keine Modalitäts-Rangfolge, sondern ein **Fehlermodus**:

<!-- AUTO-TABLE:global-formtype-de -->
| Formtyp | n | Judge Faithfulness | Clarity | Completeness |
|---|---|---|---|---|
| near-flat | 20 | 3,85 | 4,20 | 5,00 |
| categorical | 28 | 4,54 | 3,71 | 5,00 |
| non-monotonic | 16 | 4,81 | 3,62 | 5,00 |
| monotonic | 8 | 5,00 | 3,88 | 5,00 |
<!-- /AUTO-TABLE:global-formtype-de -->

**Über-Attribution vernachlässigbarer Features** ist der durchgängige Fehler, konvergent
belegt über Rubric, Judge und die qualitative Analyse
(`analyses/error_examples_by_formtype.md`). Die Stratum-n sind zu klein für Tests, die
Werte sind daher deskriptiv zu lesen.

### G2b: das Gesamtmodell pro Aufruf (`04Ge`, explorativ)

Fünf Bedingungen × 2 XAI-Modelle. Jede Antwort muss ein starres
`[FEATURE: name] [EFFECT] … [IMPORTANCE] …`-Schema je Feature ausgeben;
`split_whole_model_record` zerlegt sie in **90** Per-Feature-Records, die dasselbe Rubric
und derselbe Judge unverändert bewerten. Ein Feature, das die Antwort auslässt, zählt als
Miss, damit wird **Coverage** direkt messbar (die Sorge aus dem Meeting: „was, wenn nur
5 von 9 stimmen?").

<!-- AUTO-TABLE:global-whole-de -->
| Bedingung | Coverage (ebm · xgb) | Judge Faithfulness | Faithfulness (OpenAI) | Completeness |
|---|---|---|---|---|
| json_all | 9/9 · 9/9 | 4,72 | 3,39 | 3,61 |
| vision_all | 9/9 · 9/9 | 4,61 | 3,44 | 3,39 |
| tooluse_all | 9/9 · 9/9 | 4,83 | 3,50 | 3,89 |
| json_beeswarm | 9/9 · 9/9 | 3,61 | 3,44 | 3,72 |
| vision_beeswarm | 9/9 · 9/9 | 3,94 | 2,83 | 3,78 |
<!-- /AUTO-TABLE:global-whole-de -->

Zwei Achsen, die der Per-Feature-Track nicht öffnen kann:

- **Achse 1, Repräsentation:** alle 9 Kurven/Plots (`*_all`) gegen den einen Beeswarm
  (`*_beeswarm`). Beide tragen *unterschiedliche* Information, werden daher **pro
  GT-Feld** bewertet. `vision_all` gibt auf **beiden** XAI-Armen alle 9 Ränge an, alle
  korrekt (`fair_total` 0,926 auf EBM, 0,889 auf XGB). Die Antworten zitieren die im
  Plot-Titel gedruckte Importance (XGB `holiday`: *„least important feature in the model
  (importance 0.012)"*). Auf dem XGB-Arm wäre die vertikale Skala im Plot ein
  irreführender Rang-Proxy (Spearman ρ = 0,23 gegen den Ground-Truth-Rang; `holiday` hat
  die zweitgrößte Streuung, aber Rang 9); die Antworten folgen dem gedruckten Wert
  stattdessen. Wo überhaupt ein Rang angegeben wurde, ist er in jeder
  Whole-Model-Bedingung exakt richtig (**90 / 90** über die fünf Bedingungen auf beiden
  XAI-Armen). Die verbleibende Differenz zwischen `vision_all` und `json_all` auf XGB
  sitzt im Structure-Feld (0,722 vs. 0,889), nicht im Rang.
- **Achse 2, Mechanismus (push vs. pull):**
  `tooluse_all` (pull) und `json_all` (numerisches push) liegen auf dem fairen Aggregat
  auf beiden XAI-Modellen innerhalb von 0,05, und kein push–pull-Vergleich über die vier
  (push, Arm)-Zellen überschreitet 0,05:

  | Push-Bedingung | XAI-Modell | pull | push | pull − push |
  | --- | --- | --- | --- | --- |
  | `json_all` | ebm | 0,972 | 0,972 | 0,000 |
  | `json_all` | xgb | 0,926 | 0,963 | −0,037 |
  | `vision_all` | ebm | 0,972 | 0,926 | +0,046 |
  | `vision_all` | xgb | 0,926 | 0,889 | +0,037 |

  „Pull ≈ numerisches push" ist damit die ehrliche Lesart bei Whole-Model-Komplexität;
  „Pull-Architektur" als Designerwartung führen, nicht als Schlagzeile.
  (`utils.global_eval.axis2_mechanism_pairwise`)
- **Beeswarm-Lesbarkeit:** `json_beeswarm` und `vision_beeswarm` sind bewusst
  **informations-gematcht** (Rang + Farbrichtung + grobe Streuung, **keine** Kurve pro
  Wert), sodass die Differenz den reinen Modalitätseffekt isoliert. Auf dem fairen
  Aggregat liest das LLM das Bild im Wesentlichen so gut wie die äquivalenten Zahlen
  (`json_beeswarm` 1,000 / 0,972 gegen `vision_beeswarm` 0,917 / 0,972).

### Prozesskennzahlen (G2a)

Direkt aus den persistierten Generierungs-Records:

<!-- AUTO-TABLE:global-process-de -->
| Form | Ø Input-Tokens | Ø Output-Tokens | Ø Latenz | Ø Tool-Calls |
|---|---|---|---|---|
| Template | 0 | 0 | 0,0 s | n/a |
| JSON | 82 421 | 720 | 18,2 s | n/a |
| Vision | 1 554 | 704 | 17,8 s | n/a |
| Tool Use | 69 228 | 901 | 25,0 s | 3,1 |
<!-- /AUTO-TABLE:global-process-de -->

Zwei Dinge sind hier abzulesen. Erstens: **der Modalitätsvergleich ist nicht
informations-gematcht.** JSON bekommt die rohe Per-Instanz-SHAP-Streuung (~12k Punkte je
Feature), Vision nur den gerenderten Plot, ein Verhältnis von **53:1** bei den
Input-Tokens. Der Vergleich misst damit auch Informations*menge*, nicht nur Modalität.
G2b behebt das durch Aggregation auf ein Gitter; G2a bewusst nicht, und die Asymmetrie
wirkt in eine nützliche Richtung: JSON hatte die 53-fache Informationsmenge und schlug das
deterministische Template trotzdem nicht.

Zweitens: **Tool-Use ruft im Schnitt nur 3,1 Tools ab**, in nahezu fester Reihenfolge:
`get_feature_importances` wird in **18 von 18** Records aufgerufen (in 15 davon als erstes),
gefolgt von `get_feature_curve` und, in 16 von 18 Records, `get_feature_plot`. Die Loop
zieht also jedes Mal den vollen Rang-Kontext ab; da der Tool-Use-Prompt diese Tools bereits
„je Feature" benennt, ist das Instruction-Following und kein emergent-agentisches Suchen.

### Generierungs-Varianz (`04Gf`): gemessen, und sie ändert die Lesart von G2a

Jede G2a-Zahl beruht auf **einer** Ziehung je Zelle bei `temperature = 1.0`. `04Gf` hat
24 Zellen (die drei schwach gewichteten Features `windspeed`, `holiday`,
`weekday`, plus `hr` als Decken-Kontrolle) je dreimal neu gezogen und
jede Ziehung mit dem Rubric **und** beiden Judges bewertet.

| Instrument | Within-Cell-sd | aufzulösende Modalitätsspanne | Verhältnis |
| --- | --- | --- | --- |
| Rubric-Total | 0,089 | 0,064 | 1,4× |
| Judge-Faithfulness (Anthropic) | **0,604** | 0,333 | **1,8×** |
| Judge-Faithfulness (OpenAI) | 0,553 | 1,278 | 0,4× |

Eine einzelne Anthropic-Judge-Ziehung bewegt den Score um fast das Doppelte des gesamten
Modalitätsunterschieds. Zieht man 5 000-mal je eine Ziehung pro Zelle und berichtet Ties
explizit, treten **alle sechs möglichen Rangfolgen** auf. In den 63 % der Iterationen
ohne Gleichstand zwischen den drei LLM-Formaten führt `vision` in 54 %, `tooluse` in
39 %, `json` in 7 %, und die modale strikte Rangfolge `vision > tooluse > json` liegt
in 42 % der ungebundenen Iterationen vorn; die mittlere Spannweite je Ziehung beträgt
0,409. Ein zweiter Monte-Carlo-Lauf skaliert das auf den vollen Mean-of-18: die zehn
nicht neu gezogenen Zellen bleiben bei ihrer beobachteten Ziehung fix, nur die acht neu
gezogenen Zellen je Format werden resampled, mit einer per-Format-Standardabweichung
von 0,08–0,10, etwa einem Viertel der Mean-of-18-Modalitätsspanne in der G2a-Tabelle oben.

Zwei Konsequenzen. Der **Nullbefund wird stärker** auf dem Anthropic-Judge: „kein
nachweisbarer Modalitätsunterschied" gilt aus einem härteren Grund als bloß kleinen
Effekten, eine einzelne Ziehung kann nicht auflösen, welche von `vision` und `tooluse`
oben steht. Jede Aussage, die auf der *nominalen Spitze* aufbaut, ist dagegen ein
Artefakt der einen gezogenen Stichprobe und darf nicht in den Text; stabil ist nur `json`
am unteren Ende (in 89 % der Full-18-Iterationen letzter Platz). Die Decken-Kontrolle
verhält sich wie erwartet: `hr` zeigt beim Rubric und beim Anthropic-Judge sd = 0,000,
die Varianz sitzt ausschließlich im near-flat-Stratum. Ein Vorbehalt auf der Judge-Seite:
der OpenAI-Judge zeigt auch auf den Kontrollzellen sd = 0,385, wo Rubric und
Anthropic-Judge beide 0 liefern. Ein Teil der gemessenen Streuung ist also judge-seitig.

### Judge-Robustheit

Beide Tracks werden von zwei Vendor unter identischem Rubric bewertet. Cross-Vendor
Krippendorff-α: per-Feature Faithfulness 0,481 / Clarity 0,435 / Completeness n/a
(beide Judges vergeben auf allen 72 Records den Wert 5, der Kriterien-Deckeneffekt macht
α undefiniert, nicht null); Whole-Model 0,295 / 0,396 / 0,638. Die Reliabilität ist
**bestenfalls moderat**. OpenAI bewertet Faithfulness im Whole-Model-Track
systematisch strenger (im Mittel etwa 1,0 Punkt, 1,17 auf `vision_all`), und die
beiden Vendor ordnen die Bedingungen unterschiedlich: das Beeswarm-Paar dreht sich
um, und `json_all` fällt bei OpenAI vom zweiten auf den vierten Platz. Daher
Vergleiche **innerhalb eines Vendors** berichten, keine absoluten Niveaus.

---

## Lokaler Track (Vergleichsbasis)

Der ursprüngliche Track: Erklärungseinheit ist **eine einzelne Vorhersage** („warum bekam
diese Stunde diesen Wert?"). Er bleibt im Repository und wird weiter reproduziert, trägt
aber seit dem 30.06. nicht mehr die Kernaussage, die Aufgabe ist zu leicht, drei von vier
Pipelines sitzen bei Faithfulness am Skalenmaximum (s. Kernbefunde). Die Schritte 4 bis 6
beschreiben diesen Track.

### Schritt 4: Drei LLM-Pipelines + deterministische Baseline

Alle LLM-Pipelines verwenden `claude-sonnet-4-6` und erzeugen deutsche, dreistufige Erklärungen
(Abschnitte `[VORHERSAGE]`, `[TREIBER]`, `[EMPFEHLUNG]`) für Mitarbeitende ohne technischen Hintergrund.

#### Pipeline 04La: Template-Baseline (`04La_Template_Pipeline_Baseline.ipynb`)

Deterministischer Textbaustein-Generator, der dieselbe Dreiteilung aus denselben SHAP-/EBM-JSONs
ohne LLM-Aufruf befüllt. Beantwortet die Standard-Reviewer-Frage: *Was leistet das LLM über
einen Textbaustein hinaus?*

#### Pipeline 04Lb: JSON → Text (`04Lb_LLM_JSON_Pipeline.ipynb`)

Das LLM erhält globale Feature Importance und lokale SHAP-/EBM-Beiträge als strukturiertes JSON.

- **Eingabe:** JSON-Payload mit Metriken, Top-Features, Feature-Werten und Top-6-Beiträgen
- **System-Prompt:** Gecacht via Anthropic Prompt Caching (> 1 024 Tokens; enthält Domain-Kontext
  und Feature-Schema)
- **Besonderheit:** `build_context_string()` denormalisiert Rohwerte in Alltagssprache
  (z.B. `temp=0.68` → `~27,9 °C`) vor dem API-Aufruf

#### Pipeline 04Lc: Vision → Text (`04Lc_LLM_Vision_Pipeline.ipynb`)

Das LLM erhält den Waterfall-Plot der Instanz als base64-kodiertes PNG.

- **Eingabe:** Bild + kurzer Textprompt mit Instanz-ID, Vorhersage und tatsächlichem Wert
- **Methode:** `ask_with_images()` aus `utils/llm.py`; multimodale Anthropic API
- **Besonderheit:** Kein numerischer Zugriff auf Beitragswerte; das Modell liest Balkenlängen
  visuell ab (potenzielle Unschärfe bei kleinen Beiträgen)

#### Pipeline 04Ld: Tool-Use (`04Ld_LLM_ToolUse_Pipeline.ipynb`)

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

- **Ablauf:** Agentic loop bis `stop_reason == "end_turn"`; durchschnittlich **6,2 Tool-Calls**
  pro Erklärung (nachgerechnet über `results/pipeline06/`, n = 20)

---

### Schritt 5: Evaluation (`05_Evaluation.ipynb`, `06_Evaluation_Ichmoukhamedov.ipynb`)

#### Quantitativer Vergleich

Mittelwerte über 20 Erklärungen pro Pipeline (2 XAI-Modelle × 10 Instanzen):

<!-- AUTO-TABLE:pipeline-quant-de -->
| Pipeline | Ø Wörter | Ø Input-Tokens¹ | Ø Output-Tokens | Gesamtkosten (20 Calls) | Ø Latenz |
|---|---|---|---|---|---|
| Template | 57 | 0 | 0 | 0,00 USD | 0,0 s |
| JSON to Text | 250 | 600 | 524 | 0,17 USD | 11,5 s |
| Vision | 239 | 1 085 | 571 | 0,19 USD | 13,2 s |
| Tool Use | 403 | 5 786 | 1 268 | 0,73 USD | 33,0 s |
<!-- /AUTO-TABLE:pipeline-quant-de -->

¹ *Input-Tokens sind die abgerechneten, nicht gecachten Tokens. JSON→Text cacht den System-Prompt
(Cache-Read-Tokens, ~10 % Preis, hier nicht gezählt), daher liegt der Wert weit unter den frisch
übertragenen Bild-Tokens von Vision.* Werte aus `results/eval_summary.csv`.

> **Hinweis zur Keyword-Faithfulness:** Die ursprüngliche keyword-basierte Faithfulness-Metrik
> wurde entfernt: sie lag am Ceiling (0,94-1,0) und differenzierte nicht zwischen Pipelines.
> Die Faithfulness-Bewertung stützt sich nun auf LLM-as-Judge und die formalen RA/SA/VA-Metriken.

#### LLM-as-Judge (drei Judge-Versionen)

**v1-Scores (Sonnet, unkalibriert) 1-5 pro Kriterium, aus `results/eval_summary.csv`:**

<!-- AUTO-TABLE:judge-scores-de -->
| Pipeline | Faithfulness | Clarity | Completeness |
|---|---|---|---|
| Template | 5,00 | 4,15 | 4,80 |
| JSON to Text | 5,00 | 4,10 | 5,00 |
| Vision | 4,50 | 4,00 | 5,00 |
| Tool Use | 5,00 | 4,10 | 5,00 |
<!-- /AUTO-TABLE:judge-scores-de -->

**Judge-Versionen:**
- **v1** (Sonnet, unkalibriert): Ceiling-Effekt ~91 % der Scores = 5; zu mildes Urteil
- **v2** (Sonnet, kalibrierte Rubrik): ~73 % Scores = 5; strukturiertere Differenzierung.
  Hinweis: v2 wurde auf einem anderen (Convenience-)Sample [42, 100, 250, 500, 1337] und **ohne
  Template** erhoben, daher nicht 1:1 mit v1/v3 vergleichbar.
- **v3** (Opus 4.8, unabhängiges Modell): strengstes Urteil. Ein systematischer Offset Opus < Sonnet
  ist *konsistent mit* einem Self-Preference-Bias, aber **nicht** dessen Beweis.
- **Cross-Vendor (erledigt):** Der Cross-Vendor-Judge ist **gelaufen**, für beide Tracks und in
  voller Breite, lokal (`results/eval_llm_judge_openai.json`, α in
  `results/eval_krippendorff_alpha.csv`) und global (`results/global_judge_openai/` 72 Records,
  `results/global_whole_judge_openai/` 90 Records). Er ist damit kein offener Punkt mehr, sondern
  ein Ergebnis: Faithfulness-α 0,481 (G2a) bzw. **0,295** (G2b), OpenAI im Whole-Model-Track systematisch
  strenger (im Mittel etwa 1,0 Punkt), und die beiden Vendor ordnen die Bedingungen
  **unterschiedlich** (das Beeswarm-Paar dreht sich um, `json_all` fällt bei OpenAI vom
  zweiten auf den vierten Platz). Details im Abschnitt „Judge-Robustheit" des globalen Tracks.

**Tool-Use-Kontext für den Judge:** v3 (Opus) erhält das vollständige Tool-Call-Transkript
(Aufrufe + Ergebnisse) als Teil von `ground_truth`, sodass per Tool abgerufene Zahlen
(PD-Kurven, Kontrafaktika, Perzentile) verifizierbar sind (Fix aus Plan-Phase 0).

#### Ichmoukhamedov-Faithfulness (`06_Evaluation_Ichmoukhamedov.ipynb`)

Formale Faithfulness-Metriken nach Ichmoukhamedov et al. (2024), n = 10 Instanzen
(Präzisions-artige Metriken; Selection-Bias siehe NB 06 §4.1):

<!-- AUTO-TABLE:faithfulness-de -->
| Pipeline | RA (Rank) | SA (Sign) | VA (Value) |
|---|---|---|---|
| JSON to Text | 1,000 | 1,000 | 1,000 |
| Tool Use | 0,988 | 1,000 | 1,000 |
| Vision | 0,846 | 1,000 | 1,000 |
<!-- /AUTO-TABLE:faithfulness-de -->

---

### Schritt 6: Fehlertaxonomie und Prompt-Fix (eingefrorene Diagnostik, lokal, nicht im Repository)

Die 30 Erklärungen mit der niedrigsten Faithfulness wurden manuell einer Fehlertaxonomie
zugeordnet, die echte Erklärungsfehler (z.B. `yr`-Vorzeichenfehler, Rangtausch bei
nahen Beiträgen) von Extraktor-Artefakten trennt. Die beiden dominanten Erklärungs-Fehlerklassen
(yr-Vorzeichen, Rangordnung) wurden direkt in den Generierungs-Prompts gefixt. Der Hauptlauf
nutzt also bereits die korrigierten Prompts. Dies war eine **eingefrorene Diagnostik**,
die diese Fixes motiviert hat; die Learnings stecken jetzt in den Prompts
(`pipeline_04/05/06`, `judge_system`) und sind durch Regressionstests abgesichert
(`tests/test_prompt_golden.py`). Das Notebook selbst ist daher
nicht mehr Teil der getrackten Pipeline.

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

Die Suite umfasst Sampling-Determinismus, Generierungs-Loop-Persistenz/Resume, Judge-Parsing-Robustheit, Statistikfunktionen, Denormalisierungs-Konsistenz, README-Konsistenz und **Prompt-Fix-Regression**. Für den globalen Track zusätzlich: der konstruierte EBM-Beeswarm, die globalen Kurven-Artefakte, die Ground-Truth-Ableitung, das deterministische Rubric, die Whole-Model-Payloads und der Forced-Schema-Splitter sowie die **Truncation-Guards** (ein Record, der ans Output-Token-Limit gestoßen ist, darf weder gespeichert noch bewertet werden).
Der Prompt-Regressionstest (`test_prompt_golden.py`) friert die SHA-256-Hashes und kritischen Constraint-Phrasen aller drei Pipeline-Prompts ein (Vorzeichen- und Rangtreue für `yr=0`, Phase-3-Fix).
Er ist ein hartes Gate: ein frischer Generierungslauf darf erst starten, wenn alle Tests grün sind.

Der README-Konsistenztest gehört zu diesem Gate: jede numerische Tabelle in beiden READMEs ist in `<!-- AUTO-TABLE:name -->`-Sentinels gefasst und wird von `utils/update_readme_tables.py` aus `results/` regeneriert. Eine Zahl, die von ihrem Artefakt abweicht, lässt die Suite rot werden, genau das verhindert, dass wieder zwei Judge-Generationen in einem Dokument stehen.

**Test-Status:** `pytest tests/` → **358 passed** (2026-09-30, Python 3.13, Commit `TBD`, nach dem finalen Commit den Hash hier eintragen).

**Wenn ein Prompt absichtlich verbessert wird:**
1. Prompt-Datei bearbeiten.
2. Neuen Hash berechnen: `shasum -a 256 prompts/<datei>.md`
3. `GOLDEN_HASHES` in `tests/test_prompt_golden.py` aktualisieren.
4. Falls sich eine Constraint-Phrase geändert hat, auch `REQUIRED_PHRASES` anpassen.
5. `pytest tests/test_prompt_golden.py` muss grün sein.

### Ausführungsreihenfolge

```
01_Data_Preprocessing        → data/train.csv, data/test.csv
02a_Modeling_AllOptions      → models/*.pkl
02b_Comparison               → results/model_comparison_summary.csv
03_Explanations_Generation   → explanations/*.json, explanations/plots/{,global/}*.png,
                               explanations/global_curve_*.json

# --- globaler Track (Haupttrack) ---
04Ga_Global_Template_Baseline → results/global/template_*.json
04Gb_LLM_JSON_Pipeline        → results/global/json_*.json
04Gc_LLM_Vision_Pipeline      → results/global/vision_*.json
04Gd_LLM_ToolUse_Pipeline     → results/global/tooluse_*.json
04Ge_Global_WholeModel        → results/global_whole/*.json,
                                results/global_whole_split/*.json
05G_Evaluation                → results/global_rubric.csv,
                                results/global_judge{,_openai}/*.json
05Gb_WholeModel_Eval          → results/global_whole_judge{,_openai}/*.json

# --- lokaler Track (Vergleichsbasis) ---
04La_Template_Pipeline_Baseline → results/pipeline00/*.json
04Lb_LLM_JSON_Pipeline          → results/pipeline04/*.json
04Lc_LLM_Vision_Pipeline        → results/pipeline05/*.json
04Ld_LLM_ToolUse_Pipeline       → results/pipeline06/*.json
05_Evaluation                → results/eval_*.{csv,png,json}
06_Evaluation_Ichmoukhamedov -> results/eval06_ichmoukhamedov/
```

> Die eingefrorene Fehlertaxonomie (Schritt 6) liegt lokal unter `notebooks/_archive/`
> und ist via `**/_archive/` aus dem Repository ausgenommen.

Der lokale Track läuft auf dem n=20-Validitäts-Sample (10 Instanzen × 2 XAI-Modelle), der
globale auf allen 9 Features × 2 XAI-Modellen.

Jedes LLM-Notebook ist durch ein `RUN_API`-Flag abgesichert (Default `False`): der gesamte
Nicht-API-Pfad (Prompt-Assemblierung, Payload-Bau, Splitter, Coverage, Rubric) wird gegen
einen deterministischen Stub verifiziert, der **nichts** nach `results/` schreibt. Erst
`True` löst den abgerechneten Lauf aus. Die Generierung ist resumable: eine bereits
existierende Ausgabedatei wird nie neu erzeugt, ein abgebrochener Lauf setzt dort fort, wo
er stehen geblieben ist.

---

## Kernbefunde

> **Status:** Alle Zahlen hier stammen aus den auto-generierten Tabellen oben. Frühere
> README-Fassungen zitierten einen überholten Judge-Lauf (Faithfulness 4,40 / 4,35 / 3,80,
> Clarity ≥ 4,55), diese Werte sind **ungültig** und wurden ersetzt.

### Globaler Track (trägt die Aussage)

1. **Der Template-Befund hängt am Judge-Vendor.** Bei der eng definierten
   Einzel-Feature-Beschreibung (G2a) ist auf dem Anthropic-Judge kein Unterschied
   zwischen Template und den drei LLM-Formaten nachweisbar (`vision` 4,61 führt
   nominal, Template 4,50, `tooluse` 4,44, `json` 4,28; alle sechs Paarvergleiche
   `p_adj ≥ 0,38`, fünf von sechs Cliff's *d* „negligible"). Auf dem OpenAI-Judge liegt
   das Template vor allen drei LLM-Formaten; template–`json` erreicht `p_adj = 0,030`
   bei δ = 0,57, hängt aber an einem einzigen Urteil und übersteht die Korrektur über
   beide Judges nicht (0,061). Zur Whole-Model-Aufgabe (G2b) wurde kein
   `template_all` gefahren, und pull liegt dort innerhalb von 0,05 zu numerischem
   push; die Aussage „LLM schlägt Template bei komplexeren Aufgaben" lässt sich mit
   diesen Daten daher nicht treffen.

2. **Über-Attribution vernachlässigbarer Features ist der durchgängige Fehlermodus.**
   Judge-Faithfulness nach Formtyp: near-flat **3,85** (n = 20) ≪ categorical 4,54 (n = 28)
   < non-monotonic 4,81 (n = 16) < monotonic 5,00 (n = 8). Konvergent belegt über Rubric, Judge
   und qualitative Fehleranalyse, der robusteste inhaltliche Befund. Genau an der
   near-flat-Grenze ist auch die Klassenzuordnung am wenigsten robust, was den Befund erklärt
   und zugleich limitiert.

3. **„Pull ≈ numerisches push" auf dem Whole-Model-Track.** `tooluse_all` (pull) und
   `json_all` (numerisches push) liegen auf dem fairen Aggregat auf beiden XAI-Modellen
   innerhalb von 0,05 (EBM 0,972 / 0,972; XGB 0,926 / 0,963); kein push–pull-Vergleich
   über die vier (push, Arm)-Zellen überschreitet 0,05. Wo überhaupt ein Rang angegeben
   wurde, ist er in jeder Whole-Model-Bedingung exakt richtig (**90 / 90** über die fünf
   Bedingungen auf beiden XAI-Armen). „Pull-Architektur" als Design*erwartung* führen,
   nicht als Schlagzeile (`limitations.md` 4.4, `axis2_mechanism_pairwise`).

4. **Der Whole-Model-Track bricht den Completeness-Ceiling.** Im Per-Feature-Track ist
   Completeness konstant 5,00 (α undefiniert, nicht informativ); im Whole-Model-Track variiert
   sie und ist cross-vendor am reliabelsten von allen drei Kriterien (α = **0,638**). Der Ceiling war ein Artefakt der
   *einfachen* Aufgabe, keine Eigenschaft der Metrik.

5. **Judge-Robustheit ist cross-vendor abgesichert, aber moderat.** Faithfulness-α 0,481 (G2a)
   bzw. **0,295** (G2b) liegen unter 0,667; OpenAI bewertet Faithfulness im Whole-Model-Track
   systematisch strenger (im Mittel etwa 1,0 Punkt, 1,17 auf `vision_all`), und die beiden
   Vendor ordnen die Bedingungen **unterschiedlich** (das Beeswarm-Paar dreht sich um, und
   `json_all` fällt bei OpenAI vom zweiten auf den vierten Platz). Daher **Vergleiche
   innerhalb eines Vendors** berichten, keine absoluten Niveaus.

### Lokaler Track (Vergleichsbasis, n = 20 je Pipeline)

> **Status dieser Befunde:** deskriptiv/explorativ. Ohne Repeated Sampling und ohne
> Inferenzstatistik sind die Unterschiede **nicht** statistisch abgesichert (siehe
> Limitationen-Tabelle in `05_Evaluation.ipynb` §7).

6. **Faithfulness trennt die Pipelines kaum.** Template, JSON→Text und Tool-Use liegen alle bei
   5,00, Vision bei 4,50 (als einzige mit Streuung, sd 0,69). Drei Pipelines am Skalenmaximum:
   Faithfulness ist auf dieser Aufgabe **am Ceiling** und kann nicht ranken.

7. **Vision ist die einzige Pipeline mit Informationsverlust.** Passend zur formalen
   Rank-Agreement (0,846 gegen 1,000 bei JSON→Text und 0,988 bei Tool-Use). Sign- und
   Value-Agreement liegen überall bei 1,000. Visuelles Ablesen von Balkenlängen ist strukturell
   ungenauer als numerischer Zugriff.

8. **Clarity differenziert ebenfalls nicht** (4,00–4,15 über alle vier); die Template-Baseline
   fällt nur bei Completeness leicht ab (4,80 gegen 5,00, dünnere Empfehlung).

9. **JSON→Text** ist die effizienteste LLM-Pipeline (≈ 0,009 USD/Erklärung, niedrigste Latenz
   11,5 s), weil System-Prompt-Caching die abgerechneten Input-Tokens niedrig hält.
   **Tool-Use** liefert die längsten, belegten Erklärungen (403 gegen 250 Wörter, +61 %) bei
   ~4,3× Kosten und ~2,9× Latenz, im Schnitt 6,2 Tool-Calls.

10. **Dieser Ceiling ist der Grund für den globalen Track.** Eine Einzelinstanz-Erklärung ist
    eine leichte Aufgabe, sie nennt drei Treiber, und jede numerische Pipeline trifft sie. Die
    Unterschiede, um die es der Arbeit geht, zeigen sich erst, wenn die Aufgabe schwerer wird:
    eine ganze Feature-Beziehung oder das Gesamtmodell zu beschreiben.
