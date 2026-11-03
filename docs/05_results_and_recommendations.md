# 05 · Results and Recommendations

## Conclusion

A **Gradient Boosting** model, trained on IPF meets before 2024 and tested on meets from 2024 onward, gives every planned
attempt a **trustworthy chance of success** (predicted and actual make rates within 3 points in every range). Flagging only the
**riskiest 20% of attempts covers 47% of all misses**; random flags would cover 20%.

## Final model

| Item | Choice |
| --- | --- |
| Model | `GradientBoostingClassifier` (300 trees, depth 5, learning rate 0.05, subsample 0.8) |
| Features | 20 inputs known before the attempt (attempt, this meet so far, lifter history, profile, meet); median + missing flag for numbers, one-hot for categories |
| Imbalance | No resampling; warning threshold tuned to **P(make) < 0.736** on training out-of-fold predictions |
| Validation | Train: 300,000 attempts from meets before 2024. Test: 200,000 attempts from meets in 2024–2026, used once. Tuning: 3-fold `StratifiedKFold` on a 100,000-attempt part of the training sample |

## Model comparison (test set, 200,000 attempts from 2024 onward)

| Model | Threshold | ROC-AUC | Balanced accuracy | Recall (missed) | Precision (missed) | F1 (missed) |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (always "made") | 0.50 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.50 | 0.764 | 0.587 | 0.221 | 0.569 | 0.318 |
| Decision Tree | 0.50 | 0.765 | 0.612 | 0.289 | 0.555 | 0.380 |
| Random Forest | 0.50 | 0.775 | 0.612 | 0.282 | 0.578 | 0.379 |
| Gradient Boosting | 0.50 | 0.776 | 0.618 | 0.297 | 0.575 | 0.391 |
| **Gradient Boosting, tuned threshold** | **0.736** | **0.776** | **0.708** | **0.656** | **0.435** | **0.523** |

- Gradient Boosting ranks attempts best (ROC-AUC 0.776), just ahead of Random Forest (0.775). The ensembles gain about 0.01 over Logistic Regression: most of the signal sits in a few strong features.
- With the tuned threshold the model warns before **2 out of 3 misses**, and 44 of every 100 warnings are real misses (the test meets' miss rate is 22%).
- Accuracy is left out on purpose: the baseline scores 78.1% while warning about nothing.
- Cross-validated and test scores are close for every model, so the results hold on a later period than the training data.

## Value for evaluators and coaches

| Attempts flagged (riskiest first) | Share of all misses caught | Miss rate among flagged |
| --- | --- | --- |
| Riskiest 5% | 14.9% | 65.3% |
| Riskiest 10% | 26.8% | 58.8% |
| Riskiest 20% | 46.6% | 51.1% |
| Riskiest 30% | 61.9% | 45.3% |

| Predicted chance | Share actually made (test) |
| --- | --- |
| 0.3–0.4 | 36.1% |
| 0.5–0.6 | 57.2% |
| 0.7–0.8 | 76.2% |
| 0.9–1.0 | 94.3% |

## Imbalance handling

Class weights, random oversampling and SMOTE **did not change ROC-AUC** (Logistic Regression 0.761–0.763, Random Forest
0.768–0.769 in cross-validation). They only shift the recall/precision trade-off, which the threshold controls directly,
without synthetic rows and without distorting the probabilities.

## What drives a made attempt

| Driver | Evidence |
| --- | --- |
| Attempt number | 52% of feature importance; third attempts: bench 46.5%, deadlift 57.9%, squat 63.1% made; Logistic Regression odds × 0.42 vs openers |
| Weight vs earlier best | Partial dependence about 0.72 up to 95% of the earlier best, 0.67 at 100%, 0.60 at 110%; odds × 1.63 at 85–90% vs 100–102.5% |
| Lift | Deadlift 0.76, squat 0.68, bench 0.61 (partial dependence); deadlift odds × 3.32 vs bench |
| Previous attempt missed | 50.1% vs 70.9% after a make; odds × 0.68 |
| Jump size | 4–8% jumps score best (0.68); no or tiny jumps lowest (0.61), because they signal a hard previous attempt |
| Earlier make rate | 0.60 for lifters at 50% → 0.73 at 90%+ |
| Misses earlier in the meet | 0.695 with none → 0.61 after two → 0.53 after five |
| Weight ÷ bodyweight | 0.70 at 1× → 0.60 at 3.5× |
| Equipment | Raw 77.9% vs single-ply 69.0% made; odds × 0.84 for single-ply |

## Recommendations

1. **Lifters and coaches: treat the earlier best as the line.** In the squat example, third attempts from 100% to 105% of the earlier best sit at 64–67%, and 107.5% drops to 49%. Use the score to see the risk of each option, next to the coach's view of how the last attempt moved.
2. **Openers are the cheapest insurance.** Openers are made 87–95% of the time, and a miss on any attempt lowers the next one to about 50%. Choosing a safe opener protects the whole lift.
3. **Plan the bench third attempt most carefully.** It is made less than half the time (46.5%), the lowest of any attempt.
4. **A bad day carries over.** Each miss earlier in the meet lowers the chance on the next lift; after early misses, a more conservative next attempt is the safer plan.
5. **Evaluators: watch the riskiest attempts.** Ranking the upcoming attempts by score and paying extra attention to the riskiest 20% covers almost half of all misses. This is where a computer-vision judging aid would help most.
6. **Use Logistic Regression to explain, Gradient Boosting to score.** The odds ratios tell a clear story for a presentation; the boosted model gives the best ranking and trustworthy probabilities.

## Limitations

- **No movement data:** the model cannot see how the previous attempt moved, so it reads a tiny jump or a repeated weight as a sign of struggle. Its output says how often *similar* attempts were made, not what happens if the weight is changed.
- **Moderate ranking power:** ROC-AUC 0.776 means the model is a clear help but far from certain; many misses come from things the results sheet does not record (technique faults, commands, depth).
- **Time drift:** the model was trained on older meets and is slightly pessimistic on recent ones (actual make rates 0.5–3 points above the predictions).
- **Sampled training data:** 300,000 of 4.9 million training attempts were used to keep run times to minutes.
- **Scope:** IPF-affiliated meets with attempt-by-attempt results only.

## Next steps

- Retrain on a larger sample (or all attempts) and check whether the ranking improves.
- Retrain each year so the probabilities follow the drift in make rates.
- Add the coach's rating of the previous attempt (or video-measured bar speed) as a feature, which addresses the jump-size limitation.
- Build the presentation: problem, data, EDA, models, value for evaluators, drivers, recommendations, limitations.
- Start the computer-vision part (squat depth and lockout checks with a pre-trained pose model) and combine both in one app.
