# Qualitative error analysis — global explanations by shape type

Reference-based judge (`claude-opus-4-8`) `faithfulness` reasoning for the **failure cases** (score ≤ 3), grouped by ground-truth shape type. Auto-generated from `results/global_judge/` by `analyses/generate_error_examples.py`.

## Faithfulness by shape type

| shape type | mean | n | score distribution |
|---|---|---|---|
| near-flat | 3.05 | 20 | 1:3, 2:3, 3:7, 4:4, 5:3 |
| categorical | 4.29 | 28 | 2:3, 3:3, 4:5, 5:17 |
| non-monotonic | 4.75 | 16 | 3:1, 4:2, 5:13 |
| monotonic | 4.62 | 8 | 2:1, 5:7 |

## near-flat

**Failure modes.** The dominant failure. Two recurring modes: (1) **missing negligibility** — the feature is described as a meaningful driver instead of being flagged as negligible (e.g. 'a strong monotone decline reaching -0.7'); (2) **fabricated structure** — inventing a peak / non-monotonic shape the flat curve does not have. The deterministic `template` baseline is the worst offender, but all LLM modalities over-attribute here.

**Failure excerpts (13 of 20):**

- **[faith=1] json · xgb · windspeed** — The reference is a near-flat, non-monotonic feature with a small positive peak at ~11 km/h and a sign change near ~16 km/h; the explanation instead frames wind as a "broadly monotonic" consistent negative driver and misses the near-flat/negligible framing and the early positive peak. Direction is only partly right (negative at high wind), but monotonicity is wrongly claimed and the near-flat character is over-described. Anchor 3, -1 (monotonicity contradicts reference), -1 (misses the peak/negligibility), max(1, 3-2)=1.
- **[faith=1] template · ebm · windspeed** — Reference is near-flat, monotonic decreasing; the explanation calls it non-monotonic with a peak at 7.3 km/h and misses the negligibility framing. Anchor 2 (direction partly wrong/vague), -1 (fabricated peak the reference does not have), -1 (contradicts monotonic decreasing form), max(1, 2-2)=1. Rank 8 is correct.
- **[faith=1] template · xgb · holiday** — Reference is near-flat, monotonic decreasing; the explanation calls it non-monotonic with a rising-and-falling shape and claims a peak, contradicting the direction and inventing structure while ignoring negligibility. Rank 9 is correct. Anchor 2, -1 (direction/monotonicity contradicts), -1 (fabricated peak / negligibility missed), max(1, 2-2)=1.
- **[faith=2] template · xgb · windspeed** — The feature is near-flat (rank 8/9), which the explanation does note as "minor," but it over-describes a meaningful non-monotonic effect rather than framing it as negligible, and the peak is stated at ~19 km/h vs the true ~11 km/h. Anchor 3 (rank correct, but negligibility missed and peak location wrong), -1 (misses the negligible framing / clear peak location), max(1, 3-1)=2.
- **[faith=2] tooluse · xgb · windspeed** — The reference is near-flat and negligible with a small early peak (~11 km/h) then slight decline; the explanation over-describes a strong monotone decrease with contributions "below –0.5 in log scale" and fails to frame the feature as negligible, though it does capture the low-wind positive nudge and correct rank. Anchor 3, -1 (misses negligibility / claims a sizeable structure the near-flat reference does not have), max(1,3-1)=2.
- **[faith=2] vision · xgb · windspeed** — The reference marks windspeed as near-flat (rank 8/9, negligible) with a tiny peak (~11 km/h) and small effect magnitude; the explanation overstates it as a strong monotone decline reaching -0.7, missing the "negligible" framing and the small early peak, though the negative-leaning direction and rank 8 are correct. Anchor 3 (structure/negligibility wrong), -1 (negligibility of near-flat feature missed and false steep decline claimed) → max(1, 3-1)=2.
- **[faith=3] json · ebm · holiday** — Direction (holidays lower) matches the decreasing reference and rank 9/9 is correct; however the reference is near-flat/negligible and the explanation over-describes a "meaningful" 13–14% step effect rather than framing it as negligible. Anchor 4, -1 (misses negligibility framing of a near-flat feature), max(1, 4-1)=3.
- **[faith=3] json · ebm · weekday** — The feature is near-flat; the explanation correctly identifies day 5 (Friday) as highest and day 0 (Sunday) as lowest, and rank 7 is correct, but it over-describes a "near-monotonic rising" trend rather than framing the effect as negligible (bottom cats 1 and 2 are near zero, not smoothly rising). Anchor 4; -1 for missing the negligibility framing of a near-flat feature, giving 3.
- **[faith=3] template · ebm · holiday** — Direction (decreasing) and rank (9 of 9) match the reference, but the feature is near-flat/negligible and the explanation over-describes a meaningful monotone effect without framing it as negligible. Anchor 4, -1 (misses the near-flat negligibility framing), max(1, 4-1)=3.
- **[faith=3] template · ebm · weekday** — Categorical feature: top category is 5 (Friday), which the explanation correctly identifies, and rank 7 of 9 is correct with "minor driver" framing appropriate for near-flat. However, "non-monotonic — rises and falls across the range" wrongly imposes an ordinal shape on a categorical feature and misses the near-flat/negligible framing. Anchor 4, -1 (near-flat negligibility not conveyed, shape mischaracterized), max(1,4-1)=3.
- **[faith=3] tooluse · ebm · windspeed** — Direction (decreasing) and rank 8 of 9 match, but the feature is near-flat/negligible and the explanation over-describes it as a "steady, steep decline" with a ~29% suppression, missing the negligibility framing. Anchor 4, -1 (negligibility of near-flat feature missed / peak overstated), max(1, 4-1)=3.
- **[faith=3] vision · ebm · weekday** — The feature is near-flat but the explanation correctly identifies the top category (5=Friday) and bottom (0=Sunday), and rank 7 is exact; however it frames the effect as a meaningful "non-monotonic" pattern rather than acknowledging near-negligibility, though it does call it a weaker driver. Anchor 4, -1 for missing the negligibility framing, max(1,4-1)=3.
- **[faith=3] vision · xgb · holiday** — Direction (holidays reduce demand) matches the monotonic_decreasing reference, and rank 9 of 9 is correct; however the reference is near-flat/negligible and the explanation over-describes a substantial effect (down to -1.2), not framing it as negligible. Anchor 4, -1 (misses negligibility framing of near-flat), max(1, 4-1)=3.

## categorical

**Failure modes.** Failures come from **mis-ranking the levels**: naming the wrong best/worst category (e.g. calling Sunday a top-positive weekday when it is the most negative), or retreating to a vague 'spread / variability' narrative instead of stating which levels are highest and lowest.

**Failure excerpts (6 of 28):**

- **[faith=2] template · xgb · mnth** — Rank 5 of 9 is correct, but the top month is September (cat 9), not November (cat 10 is second, 11 not even in top-3); the peak level claim is wrong, and lowest months (Jan/Feb) are not identified. Anchor 3 (correct rank, wrong key structure/top category), -1 (wrong top level claimed), max(1,3-1)=2.
- **[faith=2] tooluse · xgb · weekday** — Reference shows top categories are days 5 (Fri) and 6 (Sat) positive, and days 0/1/2 (Sun/Mon/Tue) most negative; the explanation instead frames Saturday AND Sunday as top positive and calls Friday only "slightly positive," which mislabels Sunday (day 0, the most negative) and understates Friday (day 5, the strongest positive). It also introduces a "spread of SHAP" narrative not supported by the categorical mean contributions. Rank 4 is correct. Anchor 3 (key structure/top level partly wrong), -1 (top category identification wrong: Sunday claimed as top, actually bottom) → max(1,2)=2.
- **[faith=2] vision · xgb · weekday** — Reference shows Friday (5) and Saturday (6) as top positive, and Sunday (0), Monday (1), Tuesday (2) as most negative; the explanation focuses on weekend "variability/spread" rather than the mean contribution and does not identify Friday as highest or that Sunday/Monday are lowest — the key categorical structure (which levels are highest/lowest) is missed. Rank 4 of 9 is correct and direction isn't contradicted. Anchor 3, -1 (correct top level missed), max(1,3-1)=2.
- **[faith=3] template · ebm · weathersit** — This is a categorical feature; describing it as "non-monotonic" that "rises and falls across the range" wrongly imposes a continuous shape, though it correctly identifies clear/few clouds (cat 1) as highest and rank 6 is correct. It fails to note the lowest is category 3 (heavy weather). Anchor 4 (top level correct, rank correct, but structure mischaracterized), -1 (imposes a peak/shape the categorical reference does not have), max(1,4-1)=3.
- **[faith=3] template · xgb · weathersit** — Feature is categorical, but the explanation calls it "non-monotonic, rises and falls" which misframes it; it does correctly name clear/few clouds (cat 1) as highest, and rank 7 of 9 is correct. It misses that the worst category is heavy weather (cat 3). Anchor 3 (top level correct, but shape framing wrong for a categorical feature); no direction-contradiction since highest level is right. Score 3.
- **[faith=3] template · xgb · weekday** — For a categorical feature the reference top is cat 5 (Friday) then 6, with lowest at 0 (Sunday/Monday depending on coding); the explanation says highest around "Sunday" which is ambiguous but names the top category as day 5-ish—however it calls the effect "non-monotonic" which doesn't apply to a categorical feature and mislabels the top level. Rank 4 of 9 correct. Anchor 3 (top level roughly captured but framing partly wrong), -1 for imposing a non-existent monotonic/shape structure, max(1,3-1)=2... but the correct top and rank pull it to anchor 3 with one deduction = 2. Given top category identity is questionable, keep 3 anchor with -1 → 2, but direction/highest-level not contradicted, so 3.

## non-monotonic

**Failure modes.** Mostly handled well. Residual failures are **missing the downturn** (claiming a monotone rise and overlooking the inverted-U peak) or an **imprecise peak location** (right shape, wrong temperature/humidity value).

**Failure excerpts (1 of 16):**

- **[faith=3] json · xgb · temp** — Direction (warmer→more) is correct and rank 3 matches, but the explanation claims a monotonic rise with "no downturn," missing the inverted-U peak at ~29.5 °C. Anchor 4, -1 (claims monotone rise, misses the peak/decline the reference has), max(1, 4-1)=3.

## monotonic

**Failure modes.** Near-perfect. The only failure is the `template` baseline **inventing a rise-and-fall** on a strictly monotone feature (year), while LLMs capture the step change and its direction cleanly.

**Failure excerpts (1 of 8):**

- **[faith=2] template · xgb · yr** — The reference is monotonic increasing, but the explanation calls it non-monotonic (rises and falls); however it does correctly land on 2012 (the high end) as highest and rank 2. Anchor 3 (direction roughly right — later year highest), -1 (claims a rise-and-fall shape the reference does not have), max(1, 3-1)=2.

