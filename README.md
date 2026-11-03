# Powerlifting: Predicting Whether an Attempt Will Be Made

Capstone project (General Assembly Data Science, Bahrain). In IPF-affiliated powerlifting meets **24.6% of attempts are missed**,
and on the third attempt only **46.5% of bench presses** are made. Lifters choose every weight by feel. This project builds a model
that gives each planned attempt a **chance of success** before it is taken, so lifters and coaches can choose attempts with a known
risk, and meet evaluators can see which attempts are high-risk.

**Status:** stage 9 of 15 complete (Logistic Regression).

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

ROC-AUC measures ranking quality and is the main comparison metric; the others depend on the threshold.

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
│   └── 04_logistic_regression.ipynb
├── results/
│   ├── metrics.csv
│   ├── test_probas.csv
│   └── lr_odds_ratios.csv
├── src/
│   ├── clean.py
│   └── prep.py
├── requirements.txt
└── README.md
```

## Current updates

| Stage | Date | What was added |
| --- | --- | --- |
| 9. Logistic Regression | 2026-11-03 | `notebooks/04_logistic_regression.ipynb`: GridSearchCV on C, odds ratios |
| 8. Preprocessing pipeline & baseline | 2026-11-03 | `src/prep.py` (date split, preprocessors, metrics) and `notebooks/03_preprocessing_and_baseline.ipynb`: DummyClassifier baseline, random vs date split check |
| 7. Course fit & limitations | 2026-11-03 | `docs/04_course_fit_and_limitations.md`: Lightning Talk checklist, scope vs the course, limitations, open questions |
| 6. Preprocessing plan | 2026-11-03 | `docs/03_preprocessing_plan.md`: fixed rules, imputing, bins, encoding, scaling, split, imbalance |
| 5. Target & features | 2026-11-03 | `docs/02_target_and_features.md`: target choice, leakage decisions, engineered history features |
| 4. EDA | 2026-11-03 | `notebooks/02_eda.ipynb`: the six Lightning Talk EDA questions, make rates by feature, time drift, jump-size pattern, leakage check |
| 3. Data overview | 2026-11-03 | `notebooks/01_data_overview.ipynb`: data dictionary, types, nulls, duplicates, hidden codes |
| 2. Problem statement | 2026-11-03 | `docs/01_problem_statement.md`: business problem, audience, objectives, success metrics |
| 1. Project setup | 2026-11-03 | Cleaning and reshaping rules (`src/clean.py`), README, `requirements.txt`, `.gitignore` |

**Next:** stage 10: tree models.

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
