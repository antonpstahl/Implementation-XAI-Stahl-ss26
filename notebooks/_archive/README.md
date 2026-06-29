# Archivierte Notebooks

Diese Notebooks gehören zum **Skalierungslauf (n≈200, Phase 3b)**, der bewusst
**nicht** Teil der aktiven Pipeline ist (Kosten ~$350; n=20 bleibt der berichtete
Validitätsstand). Sie sind hier zur Provenienz aufbewahrt, nicht gelöscht.

| Notebook | Zweck | Erwartete Eingabe (existiert in der aktiven Pipeline nicht mehr) |
|---|---|---|
| `07b_Scaling_Evaluation.ipynb` | Opus- + Cross-Vendor-Judge auf n≈200, Bootstrap-CIs, Varianz | `results/scale_eval/` (Scale-Generierung aus 00/04/05/06 mit altem `SCALE_RUN=True`) |
| `08b_Scaling_Faithfulness.ipynb` | RA/SA/VA auf n≈200, Batch-Extraktion | `results/scale_eval/` |

**Reaktivierung:** Der `SCALE_RUN`-Schalter wurde aus 00/04/05/06 entfernt
(Commit „Remove SCALE_RUN scale path"). Für einen erneuten Skalierungslauf
müsste der parametrisierte Generierungspfad (`scale_instance_ids()`,
`run_batch_generation`) wieder eingehängt werden — die Util-Funktionen sind in
`utils/` unverändert vorhanden.

**Cross-Vendor-Judge:** Der OpenAI-Batch-Judge (`gpt-4o-mini`) aus 07b ist als
nächster Schritt in die aktive n=20-Evaluation (NB 07) zu portieren — er kostet
praktisch nichts und schließt die in der README genannte Cross-Vendor-Lücke.
