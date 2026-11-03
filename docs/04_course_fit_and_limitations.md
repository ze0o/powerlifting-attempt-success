# 04 · Course Fit and Limitations

## Lightning Talk 1 checklist

| Requirement (guidelines + PDF) | Where it is covered |
| --- | --- |
| Problem statement | [01_problem_statement.md](01_problem_statement.md) |
| Potential audience | [01_problem_statement.md](01_problem_statement.md) |
| Goals | [01_problem_statement.md](01_problem_statement.md) |
| Proposed solution (regression / classification) | Supervised binary classification, see [01](01_problem_statement.md) and [03](03_preprocessing_plan.md) |
| Data source, how it was collected, size, features, target | [data/README.md](../data/README.md), [01_data_overview.ipynb](../notebooks/01_data_overview.ipynb) |
| Data dictionary | [01_data_overview.ipynb](../notebooks/01_data_overview.ipynb) §2 |
| Success metrics | ROC-AUC, balanced accuracy, recall / precision / F1 for missed attempts vs a `DummyClassifier` ([01](01_problem_statement.md)) |
| 15k+ rows, 10+ columns | 4,050,248 × 42 raw; 6,486,914 attempts × 20 features after reshaping |
| EDA: target and what it represents | [02_eda.ipynb](../notebooks/02_eda.ipynb) §1 |
| EDA: categorical or continuous, balanced, distribution plot | §2 |
| EDA: feature types and counts | §3 |
| EDA: nulls per feature | §4 |
| EDA: correlations with the target and with each other | §10 |
| EDA: outliers | §5 |

## Fit with the course

Everything in this project comes from the supervised ML lessons (01–17).

| Area | Course lessons | Used for |
| --- | --- | --- |
| EDA: distributions, IQR outliers, missing values, Pearson / Spearman | 05 | Notebooks 01–02, cleaning rules |
| Binning, grouping rare categories, one-hot, scaling | 06 | `src/clean.py`, `src/prep.py` |
| Train/test split, cross-validation, data leakage | 04, 07 | Split by date, leakage check on result columns |
| `DummyClassifier` baseline | 08 | Notebook 03 |
| Logistic Regression, odds ratios | 09 | Notebook 04 |
| Confusion matrix, precision, recall, F1, ROC-AUC, threshold | 10 | Every model; notebook 07 |
| `class_weight`, `RandomOverSampler`, `SMOTE` | 11 | Notebook 06 |
| Regularization strength `C` | 12 | Notebook 04 |
| `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `GridSearchCV`, `RandomizedSearchCV`, `StratifiedKFold` | 13 | Leak-free preprocessing and tuning |
| Decision Tree | 14 | Notebook 05 |
| Random Forest, Gradient Boosting | 15 | Notebook 05 |
| Feature importance, partial dependence, SHAP | 16 | Notebook 08 |

**Note on the split:** the course splits with `train_test_split`. Here the split is a plain pandas filter on the meet date
(train before 2024, test from 2024), for two reasons found in the EDA: a lifter's attempts at one meet are linked (a random
split would put their 2nd squat in train and their 3rd in test), and the make rate drifts over time. Cross-validation inside
the training set still uses the course's `StratifiedKFold`.

**Note on sampling:** models train on a 300,000-attempt sample and are tested on a 200,000-attempt sample, so every
notebook runs in minutes. With millions of rows available, a bigger sample changes the scores very little (to be checked
in the next steps).

Left out because the course does not teach them, or they don't scale here: probability calibration tools
(replaced by a predicted-vs-actual table with `pd.cut`), group-based cross-validation (replaced by the date split),
KNN and SVM (too slow for hundreds of thousands of rows).

## Limitations

- **No movement information:** the data shows *what* happened to each attempt, not *how* the bar moved. A slow, grinding second
  attempt and a fast one look the same, which explains the "small jump, more misses" pattern (notebook 02 §9). Video would add this.
- **Correlation, not advice:** the model says how often attempts *like this one* were made, not what would happen if the lifter
  changed the weight. Lifters who pick small jumps are often the ones struggling.
- **History only inside OpenPowerlifting:** 26.5% of attempts are at a lifter's first recorded meet, so their history features are empty.
- **Missing attempts data:** 36.9% of IPF rows report only best lifts, not attempts; those meets are not in the training data.
- **Time drift:** make rates rose from 74.6% (before 2024) to 78.1% (2024 onward); a model trained on older meets is slightly
  pessimistic until retrained.
- **One rulebook:** IPF-affiliated meets only; other federations (different equipment and judging) are out of scope.
- **Same-name lifters:** two different lifters with the same name and no `#` number would share a history.

## Open questions

- [ ] Confirm the dataset is approved by the IAs.
- [ ] Final deck: which models? Suggested: Logistic Regression (readable odds ratios), the best tree ensemble, and the baseline.
- [ ] Is the probability (for attempt selection) or the risk flag (for evaluators) the main deliverable for the presentation?
- [ ] Should a later version add the computer-vision part (bar speed of the previous attempt) as a feature?
