# Qualitative error analysis — global explanations by shape type

Reference-based judge (`claude-opus-4-8`) `faithfulness` reasoning for the **failure cases** (score ≤ 3), grouped by ground-truth shape type. Auto-generated from `results/global_judge/` by `analyses/generate_error_examples.py`.

## Faithfulness by shape type

| shape type | mean | n | score distribution |
|---|---|---|---|
| near-flat | 3.85 | 20 | 1:1, 2:2, 3:5, 4:3, 5:9 |
| categorical | 4.54 | 28 | 2:1, 3:2, 4:6, 5:19 |
| non-monotonic | 4.81 | 16 | 3:1, 4:1, 5:14 |
| monotonic | 5.00 | 8 | 5:8 |

## near-flat

**Failure modes.** The dominant failure. Two recurring modes: (1) **missing negligibility** — the feature is described as a meaningful driver instead of being flagged as negligible (e.g. 'a strong monotone decline reaching -0.7'); (2) **fabricated structure** — inventing a peak / non-monotonic shape the flat curve does not have. All four handover formats over-attribute here; the `template` baseline is not worse than the LLM formats on this stratum (n = 5 per format, so no ordering among them is read into it).

**Failure excerpts (8 of 20):**

- **[faith=1] json · xgb · windspeed** — The reference is a near-flat, non-monotonic feature with a small positive peak at ~11 km/h and a sign change near ~16 km/h; the explanation instead frames wind as a "broadly monotonic" consistent negative driver and misses the near-flat/negligible framing and the early positive peak. Direction is only partly right (negative at high wind), but monotonicity is wrongly claimed and the near-flat character is over-described. Anchor 3, -1 (monotonicity contradicts reference), -1 (misses the peak/negligibility), max(1, 3-2)=1.
- **[faith=2] tooluse · xgb · windspeed** — The reference is near-flat and non-monotonic with a small early peak (~11 km/h) and sign change ~16 km/h; the explanation describes a steady monotonic decrease and misses the small positive bump / peak, and does not frame it as negligible/near-flat despite calling it minor. Rank 8 is correct, but the monotonic-decline framing contradicts the non-monotone shape and the peak is missed. Anchor 3, -1 (monotonicity contradicts), max(1,3-1)=2.
- **[faith=2] vision · xgb · windspeed** — The reference is near-flat and non-monotonic with a small positive peak at ~11 km/h and a sign change at ~16 km/h, but the explanation describes a consistent, increasingly negative effect and misses the near-flat/negligible framing and the early positive bump. Rank 8/9 is correct. Anchor 3 (direction partly captured for high winds but structure wrong and negligibility largely under-stated), -1 (missed the near-flat/peak structure). max(1, 3-1)=2.
- **[faith=3] json · ebm · holiday** — Direction (holidays lower) matches the decreasing reference and rank 9/9 is correct; however the reference is near-flat/negligible and the explanation over-describes a "meaningful" 13–14% step effect rather than framing it as negligible. Anchor 4, -1 (misses negligibility framing of a near-flat feature), max(1, 4-1)=3.
- **[faith=3] json · ebm · weekday** — The feature is near-flat; the explanation correctly identifies day 5 (Friday) as highest and day 0 (Sunday) as lowest, and rank 7 is correct, but it over-describes a "near-monotonic rising" trend rather than framing the effect as negligible (bottom cats 1 and 2 are near zero, not smoothly rising). Anchor 4; -1 for missing the negligibility framing of a near-flat feature, giving 3.
- **[faith=3] template · ebm · weekday** — The feature is categorical/near-flat and the explanation correctly frames it as negligible with rank 7 of 9; however it applies "non-monotonic, rises and falls" which is inappropriate for a categorical feature and it names no top/bottom levels (e.g. day 5 highest, day 0 lowest). Anchor 4, -1 for the misapplied shape framing (though negligibility is captured), max(1,4-1)=3... but the negligibility framing is correct and rank is right, so anchor 4 with one deduction gives 3.
- **[faith=3] tooluse · ebm · windspeed** — Direction (decreasing) and rank 8 of 9 match, but the feature is near-flat/negligible and the explanation over-describes it as a "steady, steep decline" with a ~29% suppression, missing the negligibility framing. Anchor 4, -1 (negligibility of near-flat feature missed / peak overstated), max(1, 4-1)=3.
- **[faith=3] vision · ebm · weekday** — The feature is near-flat but the explanation correctly identifies the top category (5=Friday) and bottom (0=Sunday), and rank 7 is exact; however it frames the effect as a meaningful "non-monotonic" pattern rather than acknowledging near-negligibility, though it does call it a weaker driver. Anchor 4, -1 for missing the negligibility framing, max(1,4-1)=3.

## categorical

**Failure modes.** Failures come from **mis-ranking the levels**: naming the wrong best/worst category (e.g. calling Sunday a top-positive weekday when it is the most negative), or retreating to a vague 'spread / variability' narrative instead of stating which levels are highest and lowest.

**Failure excerpts (3 of 28):**

- **[faith=2] tooluse · xgb · weekday** — The reference shows top categories are Friday (5) and Saturday (6) with weekend/Friday positive, and bottom being Sunday (0), Monday (1), Tuesday (2). The explanation correctly identifies Sunday as most negative and rank 4, but wrongly claims Mon–Thu (1–4) are positive when the reference shows Mon (1), Tue (2) are among the lowest; it also misses that Friday (5) is the strongest positive. Anchor 3 (correct direction on Sunday and rank but key structure partly wrong), -1 for misrepresenting which levels are highest. max(1,3-1)=2.
- **[faith=3] template · ebm · weathersit** — This is a categorical feature; describing it as "non-monotonic" that "rises and falls across the range" wrongly imposes a continuous shape, though it correctly identifies clear/few clouds (cat 1) as highest and rank 6 is correct. It fails to note the lowest is category 3 (heavy weather). Anchor 4 (top level correct, rank correct, but structure mischaracterized), -1 (imposes a peak/shape the categorical reference does not have), max(1,4-1)=3.
- **[faith=3] template · xgb · weathersit** — Feature is categorical, but the explanation calls it "non-monotonic, rises and falls" which misframes it; it does correctly name clear/few clouds (cat 1) as highest, and rank 7 of 9 is correct. It misses that the worst category is heavy weather (cat 3). Anchor 3 (top level correct, but shape framing wrong for a categorical feature); no direction-contradiction since highest level is right. Score 3.

## non-monotonic

**Failure modes.** Mostly handled well. The residual failure is **missing the downturn**: claiming a monotone rise and overlooking the inverted-U peak.

**Failure excerpts (1 of 16):**

- **[faith=3] json · xgb · temp** — Direction (warmer→more) is correct and rank 3 matches, but the explanation claims a monotonic rise with "no downturn," missing the inverted-U peak at ~29.5 °C. Anchor 4, -1 (claims monotone rise, misses the peak/decline the reference has), max(1, 4-1)=3.

## monotonic

**Failure modes.** At ceiling: every format, including the `template` baseline, captures the step change and its direction.

_No failures at this threshold — all explanations scored above 3._

