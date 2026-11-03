# Powerlifting: Predicting Whether an Attempt Will Be Made

Capstone project (General Assembly Data Science, Bahrain). In IPF-affiliated powerlifting meets **24.6% of attempts are missed**,
and on the third attempt only **46.5% of bench presses** are made. Lifters choose every weight by feel. This project builds a model
that gives each planned attempt a **chance of success** before it is taken, so lifters and coaches can choose attempts with a known
risk, and meet evaluators can see which attempts are high-risk.

**Status:** stage 13 of 15 complete (Explainability).

## Problem statement

> Predict, before an attempt is taken, the probability that it will be made (a good lift), so lifters and coaches can choose
> attempts with a known risk, and meet evaluators can see which attempts are high-risk.

- **Type:** supervised binary classification
- **Target:** `made` (made = 1, missed = 0), from the official results (a negative weight = missed)
- **Audience:** lifters and coaches, meet evaluators and referees, federations
- **Metrics:** ROC-AUC (main), balanced accuracy, recall / precision / F1 for missed attempts, calibration table
  (accuracy alone is misleading: always predicting "made" scores 78.1% on the test meets)

Full details: [docs/01_problem_statement.md](docs/01_problem_statement.md)

## Dataset

[OpenPowerlifting](https://www.openpowerlifting.org/) export of 2026-10-17: **4,050,248 rows × 42 columns** (one row per lifter per meet,
1,016,942 lifters, 63,900 meets, 1964–2026). After cleaning, the IPF-affiliated meets are reshaped into **6,486,914 attempts**
(one row per squat, bench or deadlift attempt) with 20 features known before the attempt. Result columns (`Best3…Kg`, `TotalKg`,
`Place`, points) are dropped because they are calculated from the attempts (leakage). Details: [data/README.md](data/README.md)

## Key EDA findings

| Finding | Number |
| --- | --- |
| Attempts made | 75.4% |
| Third attempts made | bench 46.5%, deadlift 57.9%, squat 63.1% (openers 87–95%) |
| After a missed attempt | 50.1% made (70.9% after a make) |
| Weight vs the lifter's earlier best | 85.9% made at 90–95% → 67.2% at 100–102.5% → 61.4% at 110%+ |
| Tiny jumps | 0–2.5% jump: 55.2% vs 5–10%: 75–76% (a sign of a hard previous attempt) |
| Earlier make rate | 63.0% (≤ 50% before) → 82.2% (> 89% before) |
| Equipment | raw 77.9% vs single-ply 69.0% |
| Time drift | 74.6% before 2024 vs 78.1% from 2024 |
| Duplicate entries | 383,462 rows: same lifter and meet in two divisions |
| Leakage | best lift = 3rd attempt ⇔ 3rd attempt made (100%) |

Details: [notebooks/02_eda.ipynb](notebooks/02_eda.ipynb)

## Model results (test set: 200,000 attempts from meets in 2024–2026)

| model | threshold | roc auc | balanced accuracy | recall (missed) | precision (missed) | f1 (missed) | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (always made) | 0.50 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.781 |
| Logistic Regression | 0.50 | 0.763 | 0.587 | 0.221 | 0.569 | 0.318 | 0.792 |
| Decision Tree | 0.50 | 0.765 | 0.612 | 0.289 | 0.555 | 0.380 | 0.793 |
| Random Forest | 0.50 | 0.775 | 0.612 | 0.282 | 0.578 | 0.379 | 0.797 |
| **Gradient Boosting** | 0.50 | **0.776** | 0.618 | 0.297 | 0.575 | 0.391 | 0.798 |
| Gradient Boosting (tuned threshold) | 0.736 | 0.776 | 0.708 | 0.656 | 0.435 | 0.523 | 0.738 |

ROC-AUC measures ranking quality and is the main comparison metric; the others depend on the threshold.
Class weights, oversampling and SMOTE did not improve ROC-AUC ([06_imbalance_comparison.ipynb](notebooks/06_imbalance_comparison.ipynb)).

## What drives a made attempt (feature importance + partial dependence + SHAP)

| Rank | Driver | Effect, everything else unchanged |
| --- | --- | --- |
| 1 | Attempt number | 52% of importance; third attempts odds × 0.42 vs openers |
| 2 | Jump size | 4–8% jumps score best (0.68); no or tiny jumps lowest (0.61) |
| 3 | Lift | deadlift 0.76, squat 0.68, bench 0.61 |
| 4 | Weight vs earlier best | about 0.72 up to 95%, 0.67 at 100%, 0.60 at 110% |
| 5 | Earlier make rate | 0.60 (50%) → 0.73 (90%+) |
| 6 | Misses earlier in the meet | 0.695 (none) → 0.61 (two) → 0.53 (five) |
| – | Sex, drug-tested, number of earlier meets | under 1% of importance each |

Details: [notebooks/08_explainability.ipynb](notebooks/08_explainability.ipynb), [results/feature_importance.csv](results/feature_importance.csv)

## Attempt selection example

A raw 83 kg lifter whose best squat at earlier meets is 200 kg has just made 195 kg on his second attempt:

| Third attempt | 197.5 | 200 | 202.5 | 205 | 210 | 215 |
| --- | --- | --- | --- | --- | --- | --- |
| Chance of a make | 56% | 65% | 67% | 66% | 64% | 49% |

The low score for the smallest jump is the model reading "tiny jump" as a sign of a hard second attempt, not advice to lift heavier.
Details: [notebooks/07_threshold_and_use_cases.ipynb](notebooks/07_threshold_and_use_cases.ipynb)

## Repository structure

```text
powerlifting-attempt-success/
├── data/
│   └── README.md            (openpowerlifting.csv and the cache are not committed)
├── docs/
│   ├── 01_problem_statement.md
│   ├── 02_target_and_features.md
│   ├── 03_preprocessing_plan.md
│   └── 04_course_fit_and_limitations.md
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_and_baseline.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_tree_models.ipynb
│   ├── 06_imbalance_comparison.ipynb
│   ├── 07_threshold_and_use_cases.ipynb
│   └── 08_explainability.ipynb
├── results/
│   ├── metrics.csv
│   ├── test_probas.csv
│   ├── lr_odds_ratios.csv
│   ├── best_params.json
│   ├── imbalance_cv.csv
│   ├── threshold.json
│   ├── risk_flagging.csv
│   ├── calibration.csv
│   ├── attempt_examples.csv
│   └── feature_importance.csv
├── src/
│   ├── clean.py
│   └── prep.py
├── requirements.txt
└── README.md
```

## Current updates

| Stage | Date | What was added |
| --- | --- | --- |
| 13. Explainability | 2026-11-03 | `notebooks/08_explainability.ipynb`: feature importance, partial dependence (numeric and categorical), SHAP with `TreeExplainer`, one attempt explained |
| 12. Threshold & use cases | 2026-11-03 | `notebooks/07_threshold_and_use_cases.ipynb`: threshold chosen on training data, share of misses caught by flagging the riskiest attempts, predicted vs actual table, `attempt_options()` demo |
| 11. Imbalance comparison | 2026-11-03 | `notebooks/06_imbalance_comparison.ipynb`: no handling vs class weights vs RandomOverSampler vs SMOTE |
| 10. Tree models | 2026-11-03 | `notebooks/05_tree_models.ipynb`: Decision Tree, Random Forest, Gradient Boosting with tuned hyper-parameters, overfitting check |
| 9. Logistic Regression | 2026-11-03 | `notebooks/04_logistic_regression.ipynb`: GridSearchCV on C, odds ratios |
| 8. Preprocessing pipeline & baseline | 2026-11-03 | `src/prep.py` (date split, preprocessors, metrics) and `notebooks/03_preprocessing_and_baseline.ipynb`: DummyClassifier baseline, random vs date split check |
| 7. Course fit & limitations | 2026-11-03 | `docs/04_course_fit_and_limitations.md`: Lightning Talk checklist, scope vs the course, limitations, open questions |
| 6. Preprocessing plan | 2026-11-03 | `docs/03_preprocessing_plan.md`: fixed rules, imputing, bins, encoding, scaling, split, imbalance |
| 5. Target & features | 2026-11-03 | `docs/02_target_and_features.md`: target choice, leakage decisions, engineered history features |
| 4. EDA | 2026-11-03 | `notebooks/02_eda.ipynb`: the six Lightning Talk EDA questions, make rates by feature, time drift, jump-size pattern, leakage check |
| 3. Data overview | 2026-11-03 | `notebooks/01_data_overview.ipynb`: data dictionary, types, nulls, duplicates, hidden codes |
| 2. Problem statement | 2026-11-03 | `docs/01_problem_statement.md`: business problem, audience, objectives, success metrics |
| 1. Project setup | 2026-11-03 | Cleaning and reshaping rules (`src/clean.py`), README, `requirements.txt`, `.gitignore` |

**Next:** stage 14: results & recommendations.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Put `openpowerlifting.csv` in `data/`, or keep it in `../Data/openpowerlifting-2026-10-17/` (see [data/README.md](data/README.md)).
Run the notebooks in order. The first one that loads the attempts builds `data/attempts_ipf.pkl` (about 2 minutes and 5 GB of RAM);
after that each load takes seconds. Notebooks 03–08 share the split and preprocessing in `src/prep.py` and write their results
to `results/`. Notebook 05 takes about 30 minutes (mostly the Gradient Boosting search) and notebook 07 about 13 minutes on a
2-core machine; the others take 1–3 minutes each.
