# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:34:48 2019

@author: moritz

This file is based on the results from process_sleep_data.py
It uses the G-values and duration information that are saved in the G_df files,
assembles them to one big G_df, and calculates averages and standard errors.
Finally, errors are calculated by bootstrapping.

1. read G_df_kk for current timescale, add it to G_df
2. save complete df
3. calculate weighted average, standard deviation and number of experiments
4. remove points with too little statistical justification
5. calculate errors from bootstrap procedure
"""

# %% imports, settings
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from statsmodels.stats.weightstats import DescrStatsW
from matplotlib.lines import Line2D

plt.style.use("seaborn")
plt.rcParams.update({"axes.facecolor": "white"})

# constants
MODEL_LAG_ORDER = 5  # model order p for (V)AR model
SURROGATE = False  # if True, use the surrogate data

N_SUBSETS = 100  # number of subsets for bootstrap procedure

BAND = "alpha"

if SURROGATE:
    SURMARKER_F = "surrogate/"
    SURMARKER_N = "surrogate_"
else:
    SURMARKER_F = ""
    SURMARKER_N = ""
timescales = [1, 2, 5, 10, 15]
sleepstages = ["LS", "DS", "REM"]
signals = ["breath", "heart", "eeg_amp_o1"]

pairs = [
    ("breath", "heart"),
    ("breath", "eeg_amp_o1"),
    ("heart", "breath"),
    ("heart", "eeg_amp_o1"),
    ("eeg_amp_o1", "breath"),
    ("eeg_amp_o1", "heart"),
]
multicol = pd.MultiIndex.from_tuples(pairs, names=("caused", "causing"))
multiind = pd.MultiIndex.from_product(
    [timescales, sleepstages], names=("timescale", "sleepstage")
)
G_df = pd.DataFrame(index=multiind, columns=multicol)


# %% load and assemble G df
for group in ["YC", "EC", "OSA"]:
    for kk in range(len(timescales)):
        # 1. read G_df_kk for current timescale, add it to G_df
        G_df_kk = pd.read_csv(
            os.path.join(
                "csv",
                "G_df",
                f"G_df_{SURMARKER_N}{group}_lo{MODEL_LAG_ORDER}_ts{timescales[kk]}s.csv",
            ),
            index_col=0,
            header=[0, 1],
            converters={
                1: ast.literal_eval,
                2: ast.literal_eval,
                3: ast.literal_eval,
                4: ast.literal_eval,
                5: ast.literal_eval,
                6: ast.literal_eval,
            },
        )
        G_df_kk.drop(labels="awake", axis="index", inplace=True)

        for row in G_df_kk.index:
            for col in G_df_kk.columns:
                G_df.loc[(timescales[kk], row), col] = G_df_kk.loc[row, col]
    # routine for the case that for every data point only so many data shall be
    # used as in the data point with the least data
    # 2. save complete df to csv
    G_df.to_csv(
        os.path.join(
            "csv", "G_df", f"G_df_{SURMARKER_N}{group}_lo{MODEL_LAG_ORDER}_complete.csv"
        )
    )
    # 3. calculate weighted average, number and duration of experiments
    averages_df = G_df.applymap(
        lambda x: DescrStatsW(
            [x[ii][0] for ii in range(len(x))],
            weights=[x[ii][1] for ii in range(len(x))],
            ddof=0,
        ).mean
    )
    # 4. remove points with too little statistical justification
    for ii in averages_df.index:
        for jj in averages_df.columns:
            if len(G_df.loc[ii, jj]) <= 4:
                averages_df.loc[ii, jj] = np.nan
    # save df to csv
    averages_df.to_csv(os.path.join("csv", f"avg_df_{SURMARKER_N}{group}_{BAND}.csv"))
    # 5. calculate errors from bootstrap procedure
    # bootstrapping only needs to be applied once, later use the picklefile
    # normalize weights to add up to 1
    print("# bootstrapping started #")
    weights = G_df.applymap(lambda x: [x[ii][1] for ii in range(len(x))])
    weights = weights.applymap(lambda x: [x[ii] / sum(x) for ii in range(len(x))])
    # pick subsets with replacement, paying attention to the weights
    bootstrap_df = pd.DataFrame(index=G_df.index, columns=G_df.columns)
    for ii in G_df.index:
        for jj in G_df.columns:
            bootstrap_df.loc[ii, jj] = np.random.choice(
                a=[G_df.loc[ii, jj][kk][0] for kk in range(len(G_df.loc[ii, jj]))],
                size=(N_SUBSETS, len(G_df.loc[ii, jj])),
                replace=True,
                p=weights.loc[ii, jj],
            )
    # calculate standard deviations of each sample
    for ii in bootstrap_df.index:
        for jj in bootstrap_df.columns:
            bootstrap_df.loc[ii, jj] = [np.mean(x) for x in bootstrap_df.loc[ii, jj]]
    st_err_df = bootstrap_df.applymap(np.std)
    st_err_df.to_csv(os.path.join("csv", f"st_err_df_{SURMARKER_N}{group}_{BAND}.csv"))
    print("# bootstrapping finished #")
