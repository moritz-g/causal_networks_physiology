# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:17:23 2019

@author: Moritz

Tests the limits of Granger causality:
    - influence of s1 on s2 and s3 (different influence on both signals)
    - two modes: with_s1 and without_s1: determines whether s1 shall be
      included to the VAR model or not. Must be run in both modes to obtain
      the complete picture
"""
# %% imports, settings
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import math
import time
import os
import functions as cf

# variables
REVERSE = True  # if REVERSE< test G-causality from z-->x (instead of x-->z)
exponent = 0.5  # exponent of 1/f**(exponent) - noise
n = 20  # number of repetitions of experiment
shift2 = 2  # order of lag operator
shift3 = 4
xlist = [10 ** (-2.5 + kk / 10) for kk in range(25)] + [
    0.99
]  # highest value must be < 0
sample_size = 2 ** 15
maxlag = 5
mode = "with_s1"  # "with_s1" or "without_s1" (c.f. docstring)
fs = 1  # arbitrary
pvalues_df = pd.DataFrame(index=xlist, columns=xlist)


# %% testing
start = time.time()
start_sampling = time.time()
for ii in range(n):
    start_n = time.time()
    print("### sample {} / {}".format(ii + 1, n))
    # create original signals
    o1 = cf.create_noise(exp=exponent, sample_size=sample_size)
    o2 = cf.create_noise(exp=exponent, sample_size=sample_size)
    o3 = cf.create_noise(exp=exponent, sample_size=sample_size)
    # vary influence of s1 on s2 ('x2') and on x3 ('x3') and length of
    # time series ('sample_size') (both logarithmically), plot p value of
    # G-causality test as function of these two variables

    # create dataframe with original signals [1/f**(exponent) noises]
    timeline = pd.DatetimeIndex(
        data=pd.date_range(start=0, freq=str(fs) + "s", periods=sample_size)
    )
    # create dataframe
    my_df = pd.DataFrame({"o1": o1, "o2": o2, "o3": o3}, index=timeline[:sample_size])
    my_df["s1"] = my_df["o1"]
    for x2 in xlist:
        # vary x2
        my_df["s2"] = (1 - x2) * my_df["o2"] + x2 * my_df["o1"]
        # shift s2 by 'shift2' periods (= apply L^'shift2' operator)
        my_df["s2"] = my_df["s2"].shift(periods=shift2)
        my_df["s2"][0:shift2] = 0
        for x3 in xlist:
            # vary x3
            my_df["s3"] = (1 - x3) * my_df["o3"] + x3 * my_df["o1"]
            # shift s3 by 'shift3' periods (= apply L^'shift3' operator)
            my_df["s3"] = my_df["s3"].shift(periods=shift3)
            my_df["s3"][0:shift3] = 0
            if mode == "with_s1":
                model1 = VAR(my_df[["s1", "s2", "s3"]])
            elif mode == "without_s1":
                model1 = VAR(my_df[["s2", "s3"]])
            else:
                raise ValueError("mode must be either with_s1 or without_s1")
            results = model1.fit(maxlag)
            # save value in pvalues_df so it can be taken into account for
            # the average later
            if ii == 0:
                pvalues_df.loc[x2, x3] = []
            if REVERSE:
                pvalues_df.loc[x2, x3].append(results.test_causality("s2", "s3").pvalue)
            else:
                pvalues_df.loc[x2, x3].append(results.test_causality("s3", "s2").pvalue)
    end_n = time.time()
    print(f"time elapsed for sample no {ii + 1}: {round(end_n - start_n, 1)} seconds")
    print(
        f"total time elapsed until now: {round((end_n - start_sampling) / 60, 1)} minutes"
    )
    print(
        "estimated time remaining: "
        f"{round((end_n - start_sampling) / (ii + 1) * (n - (ii + 1)) / 60, 1)} min"
    )
# routine for saving results to txt file and create "heatmap" image
results_df = pd.DataFrame(index=xlist, columns=xlist)
for row in pvalues_df.axes[0]:
    for col in pvalues_df.axes[1]:
        results_df.loc[row, col] = sum(pvalues_df.loc[row, col]) / len(
            pvalues_df.loc[row, col]
        )
if REVERSE:
    results_df.to_csv(os.path.join("csv", f"test_G_causality_x3_x2_{mode}_reverse.csv"))
else:
    results_df.to_csv(os.path.join("csv", f"test_G_causality_x2_x3_{mode}.csv"))
