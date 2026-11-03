"""Shared split, preprocessing and evaluation helpers for the model notebooks.

Every model notebook imports from here, so the train/test split and the
preprocessing are identical across all models.
"""

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_START = '2024-01-01'     # train on meets before this date, test on meets from it onward
TRAIN_N = 300_000             # random sample of training attempts (pool: about 4.9 million)
TEST_N = 200_000              # random sample of test attempts (pool: about 1.6 million)
TUNE_N = 100_000              # smaller sample of the training set used for hyper-parameter searches

TARGET = 'made'

NUMERIC = ['weight', 'weight_to_bw', 'pct_prev_best', 'jump_kg', 'jump_pct', 'Age', 'BodyweightKg',
           'n_prior_meets', 'days_since_last', 'prior_make_rate', 'attempt_no',
           'misses_this_lift', 'misses_earlier_lifts']
CATEGORICAL = ['lift', 'prev_result', 'Sex', 'Equipment', 'Event', 'tested', 'meet_country']

# The linear model gets the two strongest numeric signals as fixed bins (lesson 06), because their
# effect on the make rate goes up and then down, which a single coefficient cannot follow.
LINEAR_NUMERIC = ['weight_to_bw', 'Age', 'BodyweightKg', 'n_prior_meets', 'days_since_last',
                  'prior_make_rate', 'misses_this_lift', 'misses_earlier_lifts']
LINEAR_CATEGORICAL = CATEGORICAL + ['pct_best_bin', 'jump_bin', 'attempt_no']

FEATURES = sorted(set(NUMERIC + CATEGORICAL + LINEAR_CATEGORICAL), key=(NUMERIC + LINEAR_CATEGORICAL).index)


def split(attempts):
    """Split by meet date, then sample each side.

    Train: meets before 2024. Test: meets from 2024 onward.
    A random split would put attempts from the same lifter at the same meet into both sets
    (e.g. their 2nd squat in train and their 3rd in test), which flatters the score.
    Splitting by date also matches real use: learn from past meets, predict future ones.
    """
    train_pool = attempts[attempts['Date'] < TEST_START]
    test_pool = attempts[attempts['Date'] >= TEST_START]
    train = train_pool.sample(TRAIN_N, random_state=RANDOM_STATE)
    test = test_pool.sample(TEST_N, random_state=RANDOM_STATE)
    X_train, y_train = to_X(train), train[TARGET]
    X_test, y_test = to_X(test), test[TARGET]
    return X_train, X_test, y_train, y_test


def to_X(frame):
    """Select the model inputs; categories become plain text for the encoders."""
    X = frame[FEATURES].copy()
    for col in CATEGORICAL + ['pct_best_bin', 'jump_bin']:
        X[col] = X[col].astype(str)
    return X


def tune_sample(X_train, y_train, n=TUNE_N):
    """A fixed random part of the training set for GridSearchCV / RandomizedSearchCV."""
    X = X_train.sample(n, random_state=RANDOM_STATE)
    return X, y_train.loc[X.index]


def linear_preprocessor():
    """For Logistic Regression: impute + scale numbers, one-hot categories and bins."""
    numeric = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler())
    return ColumnTransformer([
        ('num', numeric, LINEAR_NUMERIC),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), LINEAR_CATEGORICAL),
    ], verbose_feature_names_out=False)


def tree_preprocessor():
    """For Decision Tree / Random Forest / Gradient Boosting: raw numbers + one-hot.

    Trees split on thresholds, so they need no scaling or bins. Missing numbers are filled
    with the median, and a 0/1 flag column records that the value was missing
    (e.g. a first-time lifter has no earlier best).
    """
    return ColumnTransformer([
        ('num', SimpleImputer(strategy='median', add_indicator=True), NUMERIC),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ], verbose_feature_names_out=False)


def evaluate(name, y_true, y_proba, threshold=0.5):
    """Return the project's metrics for one model as a dict.

    The positive class is 'made' (1). 'recall_missed' is the share of missed attempts the
    model flags (predicts below the threshold): the number that matters to evaluators.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        'model': name,
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_made': precision_score(y_true, y_pred, zero_division=0),
        'recall_made': recall_score(y_true, y_pred),
        'precision_missed': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        'recall_missed': recall_score(y_true, y_pred, pos_label=0),
        'f1_missed': f1_score(y_true, y_pred, pos_label=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
    }


def save_result(row, path='../results/metrics.csv'):
    """Append (or replace) one model's metrics in the shared results table."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old['model'] != row['model']]
        new = pd.concat([old, new], ignore_index=True)
    new.round(4).to_csv(path, index=False)
    return new.round(4)
