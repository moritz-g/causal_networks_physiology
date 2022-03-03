# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:05:03 2019

@author: DELL

Tests the limits of G- and PRSA-causality in terms of signal length and
influence of s1 on s2. Two signals only.

    - influence of s1 on s2: How weak can the influence of s1 be so that
      causality is still correctly detected?
    - How short can the time series become?

Tests for causality with different methods:
    0. G causality
    1. compare MPRSA with "best-fitting normal distribution"
       (one-sided Kolmogorov-Smirnov test)
    2. compare MPRSA with "best-fitting normal distribution"
       (one-sided Anderson-Darling test)
    3. compare MPRSA with empirically generated MPRSA that definitely shows
       no causality, i.e. with random anchor points
       (two-sided Kolmogorov-Smirnov test)
    4. Shapiro-Wilk normality test
"""
# %% imports, setting
from functions import create_noise, mprsa
from statsmodels.tsa.api import VAR
import pandas as pd
import numpy as np
import math
import os
from scipy import stats
import time

# parameters
len_mprsa = 15
exponent = 0.5
shift1 = 3  # order of lag operator
n = 20  # number of repetitions of experiment
xlist = [10 ** (-2.5 + kk / 40) for kk in range(100)]
sample_sizelist = [2 ** kk for kk in range(6, 17)]
tests = [
    "G-causality",
    "PRSA_KS_onesided",
    "PRSA_AD_onesided",
    "PRSA_KS_twosided",
    "PRSA_SW",
]
# %% testing
start_sampling = time.time()
counter = 0
pvalues_df = [
    pd.DataFrame(index=xlist, columns=sample_sizelist) for ff in range(len(tests))
]
for aa in range(len(tests)):
    for row in pvalues_df[aa].axes[0]:
        for col in pvalues_df[aa].axes[1]:
            pvalues_df[aa].loc[row, col] = []
while counter < n:
    counter += 1
    start_n = time.time()
    print("### sample {} / {}".format(counter, n))
    o1 = create_noise(sample_size=max(sample_sizelist), exp=0.5)
    o2 = create_noise(sample_size=max(sample_sizelist), exp=0.5)

    # 1. vary sample_size
    for sample_size in sample_sizelist:
        print("sample size: {}".format(sample_size))
        # 2. vary influence of s1 on s2
        for x in xlist:
            s1 = o1[:sample_size]
            s2 = [0] * shift1 + [
                x * o1[ii] + (1 - x) * o2[ii] for ii in range(sample_size - shift1)
            ]

            # calculate mprsa of s2 with anchor points set by s1
            results = mprsa(
                [
                    s1,
                ],
                s2,
                len_mprsa,
                crit=("inc_1",),
            )
            # df is the dataframe where all the PRSAs are saved to
            df, anchors = results
            # create random BPRSA sample to compare cdf of this with the
            # cdf of real BPRSA
            # used for two-sided tests
            df["random_MPRSA"] = mprsa(len(anchors), s2, len_mprsa)[0]["MPRSA"]
            # normalize MPRSA
            norm_loc = df["MPRSA"].mean()
            norm_scale = df["MPRSA"].std()
            df["MPRSA_normalized"] = (df["MPRSA"] - norm_loc) / norm_scale

            # method 0: G-causality
            timeline = pd.date_range(start=0, freq="s", periods=sample_size)
            my_df = pd.DataFrame({"s1": s1, "s2": s2}, index=timeline[:sample_size])
            model1 = VAR(my_df[["s1", "s2"]])
            results_G = model1.fit(4)
            normalitytest0 = results_G.test_causality("s2", "s1").pvalue
            pvalues_df[0].loc[x, sample_size].append(normalitytest0)

            # method 1: one-sided Kolmogorov-Smirnov test
            normalitytest1 = stats.kstest(rvs=df["MPRSA_normalized"], cdf="norm")
            pvalues_df[1].loc[x, sample_size].append(normalitytest1[1])

            # method 2: one-sided Anderson-Darling test
            normalitytest2 = stats.anderson(df["MPRSA_normalized"], "norm")
            if normalitytest2[0] > normalitytest2[1][3]:
                pvalues_df[2].loc[x, sample_size].append(0)
            else:
                pvalues_df[2].loc[x, sample_size].append(1)
            # if counter == 1 and x < 2**6:
            # print(pvalues_df[2].head(), pvalues_df[3].head())

            # method 3: two-sided Kolmogorov-Smirnov test
            normalitytest3 = stats.ks_2samp(df["MPRSA"], df["random_MPRSA"])
            pvalues_df[3].loc[x, sample_size].append(normalitytest3[1])

            # method 4: Shapiro-Wilk test
            normalitytest4 = stats.shapiro(df["MPRSA"])
            pvalues_df[4].loc[x, sample_size].append(normalitytest4[1])
    end_n = time.time()
    print(
        "time elapsed for sample no", counter, ":", round(end_n - start_n, 1), "seconds"
    )
    print(
        "total time elapsed until now:",
        round((end_n - start_sampling) / 60, 1),
        "minutes",
    )
    print(
        "estimated time remaining:",
        round(
            ((end_n - start_sampling) / counter * n - (end_n - start_sampling)) / 60, 1
        ),
        "min",
    )
# averaging
for aa in range(len(tests)):
    results_df = pd.DataFrame(index=xlist, columns=sample_sizelist)
    for row in pvalues_df[aa].axes[0]:
        for col in pvalues_df[aa].axes[1]:
            results_df.loc[row, col] = sum(pvalues_df[aa].loc[row, col]) / len(
                pvalues_df[aa].loc[row, col]
            )
    results_df.to_csv(os.path.join("csv", f"{tests[aa]}_x_sample-size.csv"))
end = time.time()
print("total time elapsed:", round(end - start_sampling, 1), "seconds")
