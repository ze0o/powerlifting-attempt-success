# Powerlifting: Predicting Whether an Attempt Will Be Made

Capstone project (General Assembly Data Science, Bahrain). In IPF-affiliated powerlifting meets **24.6% of attempts are missed**,
and on the third attempt only **46.5% of bench presses** are made. Lifters choose every weight by feel. This project builds a model
that gives each planned attempt a **chance of success** before it is taken, so lifters and coaches can choose attempts with a known
risk, and meet evaluators can see which attempts are high-risk.

**Status:** stage 2 of 15 complete (Problem statement).

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

## Repository structure

```text
powerlifting-attempt-success/
├── data/
│   └── README.md            (openpowerlifting.csv and the cache are not committed)
├── docs/
│   └── 01_problem_statement.md
├── src/
│   └── clean.py
├── requirements.txt
└── README.md
```

## Current updates

| Stage | Date | What was added |
| --- | --- | --- |
| 2. Problem statement | 2026-11-03 | `docs/01_problem_statement.md`: business problem, audience, objectives, success metrics |
| 1. Project setup | 2026-11-03 | Cleaning and reshaping rules (`src/clean.py`), README, `requirements.txt`, `.gitignore` |

**Next:** stage 3: data overview.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Put `openpowerlifting.csv` in `data/`, or keep it in `../Data/openpowerlifting-2026-10-17/` (see [data/README.md](data/README.md)).
Run the notebooks in order. The first one that loads the attempts builds `data/attempts_ipf.pkl` (about 2 minutes and 5 GB of RAM);
after that each load takes seconds.
