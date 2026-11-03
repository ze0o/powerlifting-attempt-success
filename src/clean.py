"""Fixed-rule cleaning and feature building shared by every notebook.

Turns the raw OpenPowerlifting file (one row per lifter per meet) into an
attempt-level table (one row per squat / bench / deadlift attempt) for
IPF-affiliated meets.

None of the steps learn anything from the data (no means, medians or fitted
values): every rule is either a fixed cut point or uses only the same lifter's
*earlier* meets. So they are safe to apply before the train/test split.
Learned steps (imputing, scaling, encoding) happen later inside a Pipeline,
on the training data only (src/prep.py).

The reasons and numbers behind every rule are in notebooks/01_data_overview.ipynb,
notebooks/02_eda.ipynb and docs/02_target_and_features.md.
"""

import os

import numpy as np
import pandas as pd

# The raw file is 828 MB, so it is not stored in the repo. Put it in data/ or keep it
# in the DS folder's shared Data/ folder; the first path that exists is used.
RAW_PATHS = ['../data/openpowerlifting.csv',
             '../../Data/openpowerlifting-2026-10-17/openpowerlifting.csv']
CACHE_PATH = '../data/attempts_ipf.pkl'

LIFTS = ['Squat', 'Bench', 'Deadlift']                    # competition order
ATTEMPT_COLS = [f'{lift}{k}Kg' for lift in LIFTS for k in [1, 2, 3]]
BEST_COLS = [f'Best3{lift}Kg' for lift in LIFTS]

USECOLS = (['Name', 'Sex', 'Event', 'Equipment', 'Age', 'BodyweightKg', 'Tested',
            'Federation', 'ParentFederation', 'Date', 'MeetCountry', 'MeetName', 'Sanctioned']
           + ATTEMPT_COLS + BEST_COLS)
CATEGORY_COLS = ['Sex', 'Event', 'Equipment', 'Tested', 'Federation', 'ParentFederation',
                 'MeetCountry', 'Sanctioned']

AGE_MIN, AGE_MAX = 10, 85          # outside this range the age is a typo
BODYWEIGHT_MIN = 30                # kg; lighter values are typos
TOP_COUNTRIES = 15                 # other meet countries are grouped as 'Other'

# Fixed bins for the linear model, chosen from the EDA (notebook 02, section 5)
PCT_BEST_BINS = [0, 85, 90, 95, 100, 102.5, 105, 110, 1000]
PCT_BEST_LABELS = ['<85', '85-90', '90-95', '95-100', '100-102.5', '102.5-105', '105-110', '110+']
JUMP_BINS = [-1000, 0, 2.5, 5, 7.5, 10, 1000]
JUMP_LABELS = ['<=0', '0-2.5', '2.5-5', '5-7.5', '7.5-10', '10+']


def raw_path():
    for path in RAW_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError('openpowerlifting.csv not found; see data/README.md')


def load_raw(usecols=USECOLS):
    """Load the columns the project needs with compact data types.

    low_memory=False reads the file in one pass (about 4-5 GB of RAM); the default
    chunked reader fails on this file in newer pandas versions.
    """
    dtypes = {c: 'float32' for c in ATTEMPT_COLS + BEST_COLS + ['Age', 'BodyweightKg'] if c in usecols}
    df = pd.read_csv(raw_path(), usecols=usecols, dtype=dtypes, parse_dates=['Date'], low_memory=False)
    cats = [c for c in CATEGORY_COLS if c in df.columns]
    df[cats] = df[cats].astype('category')
    return df


def clean(df, verbose=True):
    """Apply the fixed row rules in order and report how many rows each removes."""
    log = [('raw file', len(df))]

    def step(name, keep):
        nonlocal df
        df = df[keep]
        log.append((name, len(df)))

    step('drop unsanctioned meets', df['Sanctioned'] != 'No')
    # the same lifter at the same meet appears once per division entered (e.g. Open and Junior)
    step('keep one row per lifter per meet', ~df.duplicated(['Name', 'Date', 'MeetName', 'Federation']))

    df = df.copy()
    df.loc[(df['Age'] < AGE_MIN) | (df['Age'] > AGE_MAX), 'Age'] = np.nan
    df.loc[df['BodyweightKg'] < BODYWEIGHT_MIN, 'BodyweightKg'] = np.nan

    if verbose:
        report = pd.DataFrame(log, columns=['step', 'rows_left'])
        report['removed'] = (-report['rows_left'].diff()).fillna(0).astype(int)
        print(report.to_string(index=False))
    return df.reset_index(drop=True)


def add_history(df):
    """Add features from each lifter's EARLIER meets only (all federations).

    - n_prior_meets      : number of earlier meets
    - days_since_last    : days since the previous meet
    - prior_make_rate    : share of attempts made at earlier meets
    - prev_best_<lift>   : best successful result on that lift at any earlier meet
    """
    df = df.sort_values(['Name', 'Date']).reset_index(drop=True)
    df['lifter_id'] = pd.factorize(df['Name'])[0].astype('int32')
    by_lifter = df.groupby('lifter_id', sort=False)

    df['n_prior_meets'] = by_lifter.cumcount().astype('int16')
    df['days_since_last'] = by_lifter['Date'].diff().dt.days.astype('float32')

    for lift in LIFTS:
        best = df[f'Best3{lift}Kg'].where(df[f'Best3{lift}Kg'] > 0)   # successful bests only
        running = best.groupby(df['lifter_id'], sort=False).cummax()
        running = running.groupby(df['lifter_id'], sort=False).ffill()
        df[f'prev_best_{lift}'] = running.groupby(df['lifter_id'], sort=False).shift(1)

    att = df[ATTEMPT_COLS]
    made_here = (att > 0).sum(axis=1)
    tried_here = (att.notna() & (att != 0)).sum(axis=1)
    made_before = made_here.groupby(df['lifter_id'], sort=False).cumsum() - made_here
    tried_before = tried_here.groupby(df['lifter_id'], sort=False).cumsum() - tried_here
    df['prior_make_rate'] = (made_before / tried_before.replace(0, np.nan)).astype('float32')
    return df


def to_attempts(df):
    """Reshape one row per lifter-meet into one row per attempt (IPF meets only).

    Target: made = 1 if the attempt was successful (a negative weight in the raw file = missed).
    Only information known BEFORE the attempt is kept: the attempt itself, earlier attempts
    at this meet, the lifter's profile and their earlier meets.
    """
    ipf = df[df['ParentFederation'] == 'IPF']
    misses = {lift: (ipf[[f'{lift}{k}Kg' for k in [1, 2, 3]]] < 0).sum(axis=1) for lift in LIFTS}
    earlier_lift_misses = {'Squat': 0, 'Bench': misses['Squat'], 'Deadlift': misses['Squat'] + misses['Bench']}

    base = ['lifter_id', 'Date', 'Sex', 'Equipment', 'Event', 'Tested', 'Age', 'BodyweightKg',
            'MeetCountry', 'Federation', 'n_prior_meets', 'days_since_last', 'prior_make_rate']
    parts = []
    for lift in LIFTS:
        for k in [1, 2, 3]:
            w = ipf[f'{lift}{k}Kg']
            keep = w.notna() & (w != 0)
            part = ipf.loc[keep, base].copy()
            part['lift'] = lift
            part['attempt_no'] = k
            part['weight'] = w[keep].abs()
            part['made'] = (w[keep] > 0).astype('int8')

            if k == 1:
                part['prev_result'] = 'none'
                part['jump_kg'] = np.nan
                part['misses_this_lift'] = 0
            else:
                prev = ipf.loc[keep, f'{lift}{k - 1}Kg']
                part['prev_result'] = np.select([prev > 0, prev < 0], ['made', 'missed'], 'unknown')
                part['jump_kg'] = (part['weight'] - prev.abs()).where(prev.notna() & (prev != 0))
                earlier = ipf.loc[keep, [f'{lift}{j}Kg' for j in range(1, k)]]
                part['misses_this_lift'] = (earlier < 0).sum(axis=1)
            em = earlier_lift_misses[lift]
            part['misses_earlier_lifts'] = 0 if np.isscalar(em) else em[keep]
            part['prev_best'] = ipf.loc[keep, f'prev_best_{lift}']
            parts.append(part)

    a = pd.concat(parts, ignore_index=True)
    return add_attempt_features(a)


def add_attempt_features(a):
    """Ratios, fixed clips, fixed bins and grouped categories (all fixed rules)."""
    prev_weight = a['weight'] - a['jump_kg']
    a['jump_pct'] = (a['jump_kg'] / prev_weight * 100).clip(-20, 40)
    a['pct_prev_best'] = (a['weight'] / a['prev_best'] * 100).clip(40, 160)
    a['weight_to_bw'] = (a['weight'] / a['BodyweightKg']).clip(0, 6)
    a['days_since_last'] = a['days_since_last'].clip(0, 3650)
    a['year'] = a['Date'].dt.year.astype('int16')

    a['pct_best_bin'] = pd.cut(a['pct_prev_best'], PCT_BEST_BINS, labels=PCT_BEST_LABELS).astype(str)
    a.loc[a['pct_prev_best'].isna(), 'pct_best_bin'] = 'no history'
    a['jump_bin'] = pd.cut(a['jump_pct'], JUMP_BINS, labels=JUMP_LABELS).astype(str)
    a.loc[a['attempt_no'] == 1, 'jump_bin'] = 'first attempt'
    a.loc[(a['attempt_no'] > 1) & a['jump_pct'].isna(), 'jump_bin'] = 'unknown'

    top = a['MeetCountry'].value_counts().index[:TOP_COUNTRIES]
    a['meet_country'] = np.where(a['MeetCountry'].isin(top), a['MeetCountry'].astype(str), 'Other')
    a['tested'] = np.where(a['Tested'] == 'Yes', 'Yes', 'No')

    float_cols = ['weight', 'jump_kg', 'jump_pct', 'prev_best', 'pct_prev_best', 'weight_to_bw']
    a[float_cols] = a[float_cols].astype('float32')
    for col in ['lift', 'prev_result', 'pct_best_bin', 'jump_bin', 'meet_country', 'tested']:
        a[col] = a[col].astype('category')
    a['attempt_no'] = a['attempt_no'].astype('int8')
    a['misses_this_lift'] = a['misses_this_lift'].astype('int8')
    a['misses_earlier_lifts'] = a['misses_earlier_lifts'].astype('int8')
    return a.drop(columns=['MeetCountry', 'Tested'])


def load_attempts(rebuild=False, verbose=True):
    """Return the attempt-level IPF table, building and caching it on first use.

    Building takes about 1-2 minutes and 5 GB of RAM; later loads take seconds.
    The cache (data/attempts_ipf.pkl) is ignored by git.
    """
    if os.path.exists(CACHE_PATH) and not rebuild:
        return pd.read_pickle(CACHE_PATH)
    df = clean(load_raw(), verbose=verbose)
    df = add_history(df)
    attempts = to_attempts(df)
    del df
    attempts.to_pickle(CACHE_PATH)
    if verbose:
        print(f'{len(attempts):,} attempts saved to {CACHE_PATH}')
    return attempts
