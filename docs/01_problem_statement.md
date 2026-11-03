# 01 · Problem Statement

## Business problem

In a powerlifting meet every lifter gets **three attempts** on the squat, bench press and deadlift, and must declare each
weight before walking out. Too light, and kilos are left on the platform; too heavy, and the attempt is missed. Missing all
three attempts on one lift ends the lifter's meet. Today these choices rest on the lifter's or coach's feel.

The stakes are real: in IPF-affiliated meets **24.6% of attempts are missed**, and on the third attempt only **46.5% of bench
presses**, 57.9% of deadlifts and 63.1% of squats are made.

## Problem statement

> **Predict, before an attempt is taken, the probability that it will be made (a good lift), so lifters and coaches can choose
> attempts with a known risk, and meet evaluators can see which attempts are high-risk.**

Business problem → why it matters → what ML will do:
attempts are chosen by feel → missed attempts cost places, totals and whole meets → a classifier gives each planned attempt a
success probability from the lifter's history and what has happened so far at the meet.

## Type of problem

**Supervised binary classification.** Target: `made` (made = 1, missed = 0), taken from the official results
(a negative weight in the raw file is a missed attempt).

## Audience

- **Lifters and coaches**: choose openers and next attempts with a known chance of success.
- **Meet evaluators and referees**: see which attempts are high-risk; later combined with a computer-vision judging aid.
- **Federations and meet directors**: understand make rates by attempt, lift, equipment and division.

## Objectives

1. Build a leakage-free, attempt-level dataset from the raw meet results, using **only information known before the attempt**.
2. Beat the naive baseline. Predicting "made" for everyone is already **75.4% accurate but never warns about a miss**,
   so accuracy alone is misleading.
3. Show practical value: how many of the misses are caught when only the riskiest attempts are flagged, and whether the
   predicted probabilities can be trusted (a predicted 70% should be made about 70% of the time).
4. Explain the drivers (coefficients, feature importance, partial dependence, SHAP) and turn them into advice for attempt selection.

## Success metrics

| Metric | Why |
| --- | --- |
| ROC-AUC | How well the model ranks attempts from risky to safe, independent of the threshold (main comparison metric) |
| Balanced accuracy | Treats made and missed equally despite the 75 / 25 split |
| Recall (missed) | Share of real misses the model warns about |
| Precision (missed) | Share of warnings that really were misses |
| F1 (missed) | Balance between recall and precision for misses |
| Calibration table | Do predicted probabilities match how often attempts were actually made? |
| Confusion matrix | Shows false alarms and missed warnings |

Every model is compared against a `DummyClassifier` baseline on meets **from 2024 onward**, after training on earlier meets.
