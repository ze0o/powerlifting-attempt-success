# Presentation

The capstone deck has **16 slides with speaker notes**, built as a Claude Slides deck. To add the file itself here,
download it as PowerPoint (`.pptx`) or PDF from the deck's **Share › Export** menu and save it in this folder as
`powerlifting_attempt_success.pptx`.

## Outline

| # | Slide | Message | Numbers on the slide |
| --- | --- | --- | --- |
| 1 | Will this attempt be made? | Title: predicting the chance of a good lift before the lifter walks out | OpenPowerlifting, IPF meets |
| 2 | Every attempt is chosen by feel | The problem | 24.6% of attempts missed; 3rd bench 46.5% made; 0 of 3 = out |
| 3 | Problem statement | Predict the probability an attempt is made; three audiences | Lifters & coaches, evaluators, federations |
| 4 | From meet results to attempts | Data pipeline: raw → clean → history → attempts | 4,050,248 rows → 6,486,914 attempts; negative weight = miss |
| 5 | Only what is known before the attempt | Feature groups and the leakage check | Best squat = 3rd attempt ⇔ made (100%) |
| 6 | Third attempts are the risky ones | Make rate by lift and attempt | Openers 86.7–94.9%; 3rd: squat 63.1%, bench 46.5%, deadlift 57.9%; after a miss 50.1% |
| 7 | The old best is a barrier | Make rate by weight as % of earlier best | 85.9% at 90–95% → 67.2% at 100–102.5% → 61.4% at 110%+ |
| 8 | Smaller jumps fail more | Correlation is not causation | 3rd after a made 2nd: 49.7% (≤1.5% jump) vs 60.6% (5–7.5%) |
| 9 | Learn from past meets, test on future ones | Date split, pipelines, models, tuning | Train before 2024 (300k of 4.9M), test 2024–2026 (200k of 1.6M) |
| 10 | Gradient Boosting ranks attempts best | Model comparison | ROC-AUC 0.500 / 0.764 / 0.765 / 0.775 / **0.776**; tuned recall (missed) 0.656 |
| 11 | A risk score you can trust | Value for evaluators and coaches | Riskiest 20% hold 47% of misses; 2 of 3 misses warned; ±3 points calibration |
| 12 | Attempt number leads by far | Drivers: importance + partial dependence | 52% importance; old best 0.72 → 0.60; consistency 0.60 → 0.73 |
| 13 | Up to 5% over the best is a fair bet | Use case: choosing the 3rd squat | 200–210 kg: 64–67%; 215 kg: 49%; 197.5 kg: 56% |
| 14 | What lifters and evaluators can do | Six recommendations | docs/05 |
| 15 | The results sheet can’t see the lift | Limitations and next steps (computer vision) | docs/04 |
| 16 | Thank you, questions? | Close | Link to this repository |

Every number comes from the executed notebooks and `results/`.
