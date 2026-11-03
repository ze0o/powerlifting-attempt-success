# 02 · Target and Feature Selection

## Unit of analysis

The raw file has one row per **lifter per meet**. The model works on one row per **attempt**: each lifter-meet is reshaped into
up to 9 rows (3 lifts × 3 attempts). IPF-affiliated meets give **6,486,914 attempts** from 290,346 lifters (`src/clean.py`).

## Target

**`made`**: was the attempt successful? made = 1, missed = 0. In the raw file a missed attempt is a negative weight
(`-37.5` = missed 37.5 kg), so the label comes straight from the official results.

| Candidate target | Type | Why chosen or not |
| --- | --- | --- |
| **made (per attempt)** | binary, 75.4% made | **Chosen**: millions of labeled examples, and the decision it informs (which weight to take) happens before every attempt |
| TotalKg at the next meet | numeric | Useful later; depends on choices the lifter makes over months, not at the meet |
| Goodlift (IPF GL points) | numeric | A score of the total, so the same problem as TotalKg |
| Bomb-out (missing all 3 attempts on a lift) | binary, rare | Follows from the attempt model: three low probabilities in a row |

## Feature decisions

| Feature | Decision | Reason |
| --- | --- | --- |
| `Best3SquatKg`, `Best3BenchKg`, `Best3DeadliftKg` | **Drop** | **Leakage**: the best lift *is* the result; best = 3rd attempt ⇔ 3rd attempt made in 100% of cases (notebook 02) |
| `TotalKg`, `Place`, `Dots`, `Wilks`, `Glossbrenner`, `Goodlift` | **Drop** | **Leakage**: calculated from the results (TotalKg is empty when a lift fails; Place holds DQ) |
| Later attempts at the same meet | **Drop** | Not known yet when the attempt is chosen |
| 4th attempts | **Drop** | Record tries that don't count toward the total |
| `Name` | ID only | Links a lifter's meets to build history features; never a model input |
| `Federation` | Drop | 108 levels, overlaps with `meet_country` |
| `Division`, `AgeClass`, `BirthYearClass`, `WeightClassKg`, `State`, `MeetState`, `MeetTown`, `MeetName`, `Country` | Drop | Free text, duplicates of `Age` / `BodyweightKg`, or mostly empty |
| `weight` and `weight_to_bw` | Keep both for trees; only `weight_to_bw` for the linear model | Correlated at 0.80 |
| `jump_kg` and `jump_pct` | Keep both for trees; bins of `jump_pct` for the linear model | Correlated at 0.68; non-linear effect |
| `pct_prev_best` | Keep; bins for the linear model | Strongest numeric signal, rises then falls |
| `Age`, `BodyweightKg`, `n_prior_meets` | Keep | Weak alone (flat make rates) but may help in combination |

## Engineered features

All of them use only what is known **before** the attempt.

| Feature | How it is built |
| --- | --- |
| `prev_best` → `pct_prev_best` | Best successful result on this lift at any **earlier** meet (all federations); attempt weight as a % of it |
| `n_prior_meets`, `days_since_last` | Count of earlier meets; days since the previous one |
| `prior_make_rate` | Share of attempts made at earlier meets |
| `prev_result` | Result of the previous attempt on this lift: none (opener), made, missed |
| `jump_kg`, `jump_pct` | Increase over the previous attempt on this lift |
| `misses_this_lift` | Misses so far on this lift (0–2) |
| `misses_earlier_lifts` | Misses on lifts already finished at this meet (squat before bench before deadlift) |
| `weight_to_bw` | Attempt weight ÷ bodyweight |
| `meet_country` | Top 15 meet countries, the rest grouped as "Other" |

## Feature groups

| Group | Features |
| --- | --- |
| The attempt | lift, attempt_no, weight, weight_to_bw |
| This meet so far | prev_result, jump_kg, jump_pct, misses_this_lift, misses_earlier_lifts |
| Lifter history | pct_prev_best, n_prior_meets, days_since_last, prior_make_rate |
| Lifter profile | Sex, Age, BodyweightKg, Equipment, tested |
| Meet | Event, meet_country |
