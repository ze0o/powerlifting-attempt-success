# 03 · Preprocessing Plan

Every method below is taught in the course (lesson numbers in the last column).

## Fixed rules (before the split, `src/clean.py`)

These learn nothing from the data, so they are safe before the train/test split (same idea as the lesson 11 lab).

| Step | Rule | Why | Lesson |
| --- | --- | --- | --- |
| Unsanctioned meets | Drop (11,436 rows) | Don't count for records | 05 |
| Multi-division entries | Keep one row per lifter per meet (383,462 extra rows) | Same attempts counted twice otherwise | 05 |
| Impossible values | Age < 10 or > 85, bodyweight < 30 kg → missing | Typos | 05 |
| History features | Built from each lifter's **earlier** meets only | No information from the current meet or the future | 06, 07 |
| Reshape | One row per attempt, IPF meets only | Unit of prediction | 05 |
| Caps | `jump_pct` −20% to +40%, `pct_prev_best` 40% to 160%, `weight_to_bw` ≤ 6, `days_since_last` ≤ 10 years | Remove data-entry extremes with fixed limits | 05 |
| Bins | `pct_best_bin`: <85 / 85–90 / 90–95 / 95–100 / 100–102.5 / 102.5–105 / 105–110 / 110+ / no history; `jump_bin`: ≤0 / 0–2.5 / 2.5–5 / 5–7.5 / 7.5–10 / 10+ / first attempt | Effects rise and fall (notebook 02 §6), which one linear coefficient cannot follow | 06 |
| Rare categories | `meet_country`: top 15 + "Other" | 100+ countries, many tiny | 06 |

## Learned steps (inside a `Pipeline`, fit on training data only, `src/prep.py`)

| Feature | Linear preprocessor (Logistic Regression) | Tree preprocessor (trees, forests, boosting) | Lesson |
| --- | --- | --- | --- |
| Numeric features | `SimpleImputer(median, add_indicator=True)` + `StandardScaler` | `SimpleImputer(median, add_indicator=True)`, no scaling | 06, 13 |
| `pct_prev_best`, `jump_pct`, `jump_kg`, `weight` | Replaced by the bins (and `weight_to_bw`) | Raw values | 06 |
| `attempt_no` | One-hot | Raw (1, 2, 3) | 06 |
| lift, prev_result, Sex, Equipment, Event, tested, meet_country | `OneHotEncoder(drop='first', handle_unknown='ignore')` | `OneHotEncoder(handle_unknown='ignore')` | 06 |

The missing flag matters here: a missing `pct_prev_best` means "first meet", and a missing `jump_pct` means "opener".

## Split

- **By meet date:** train on meets **before 2024**, test on meets **from 2024 onward**. A random split would put the same
  lifter's attempts from one meet into both sets (their 2nd squat in train, 3rd in test), and the make rate has drifted from
  74.6% to 78.1%, so the date split gives the honest, "future meets" score. It is a plain pandas filter on `Date`.
- **Samples:** 300,000 training attempts (from 4.9 million) and 200,000 test attempts (from 1.6 million), so every model,
  including Gradient Boosting, trains in minutes on a laptop. Hyper-parameter searches use a 100,000-attempt part of the training sample.
- **Cross-validation:** `GridSearchCV` / `RandomizedSearchCV` with `StratifiedKFold` on the training sample only (lessons 04, 13).

## Imbalance (75 / 25)

Compare the lesson 11 options on the training data only: no handling, `class_weight='balanced'`, `RandomOverSampler`
and `SMOTE` (after encoding), wrapped in imblearn's `make_pipeline` so resampling only touches training folds. Then move
the decision threshold with `predict_proba` and `precision_recall_curve` (lesson 10).

## Models to compare

| Model | Why | Lesson |
| --- | --- | --- |
| `DummyClassifier` | Baseline: always "made" | 08 |
| Logistic Regression | Readable odds ratios | 09, 12 |
| Decision Tree | One readable tree, non-linear | 14 |
| Random Forest | Bagged trees, usually stronger | 15 |
| Gradient Boosting | Trees fixing each other's errors | 15 |

KNN and SVM (lessons 08, 17) are left out: both scale badly to hundreds of thousands of rows (KNN compares each new attempt
with every training attempt; kernel SVM training time grows with the square of the rows).

## Order of operations

1. Fixed rules, history features and reshape (`load_attempts()`).
2. Split by date, then sample.
3. Build the `ColumnTransformer` and put it in a `Pipeline` with the model.
4. Resample (if used) inside the pipeline, so it only touches training folds.
5. Tune with cross-validation on the training sample; evaluate once on the untouched test sample.
