# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:30:12 2019

@author: DELL

This file uses results from preprocess_sleep_data.py and creates csv data
containing dataframes with information about the G-values and the duration of
each sleep stage for each time scale.
Analysis should be continued using process_G_df.py

stat_df_new is used to save results from stationarity analysis approach.

Procedure:
----------
0. make list of all experiments
Then for every time scale (1, 2, 5, 10, ... seconds):
1. read sleep stages and create sleep_stage_series (list with
   beginnings and ends of all continuous sleep stages in the experiment)
2. resample time series to new time scale
3. calculate indices start_index and stop_index of new timeline at
   which every sleep stage starts and ends
4. cut patches to include only the window of the sleep stage in question
5. normalize time series (subtract mean, divide by std)
6. calculate G causality using checkStatCalcG function, write to dataframe
7. save DataFrame to G_df folder
"""

# %% imports, settings
import time
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import sys
import os

group = "YC"
# 'YC', 'EC', 'OSA', 'surrogate'
# - Young Control, Elderly Control, Sleep Apnea, surrogate
surrogate = False
# this is just to make sure all files will be named correctly. A bit dirty.
if surrogate:
    surmarker_f = "surrogate/"
    surmarker_n = "surrogate_"
else:
    surmarker_f = ""
    surmarker_n = ""
timescales = [1, 2, 5, 10, 15, 30, 60, 120]  # seconds
sleepstages = ["awake", "LS", "DS", "REM"]
band = "alpha"
signals = ["breath", "heart", "eeg_amp_o1"]
alpha = 0.05  # null hypothesis rejection level
model_lag_order = 5  # maximum VAR model order
# %% definitions


def checkStatCalcG(df_all, order, duration):
    """1. Check all time series for stationarity at model order 'order'
    2. If all stationary --> apply G calculation, write and quit
    3. If not: reduce order by 1 and try again
    4. If model order reaches 1: split patch in half and append it to
       sleep_stages_list so in a later step it can be tried again if the
       smaller pieces are stationary."""

    try:
        breath_stat_p = adfuller(
            df_all["breath"].dropna(), regression="ct", maxlag=order, autolag=None
        )[1]
        if breath_stat_p < alpha:
            breath_stat = True
        else:
            breath_stat = False
        heart_stat_p = adfuller(
            df_all["heart"].dropna(), regression="ct", maxlag=order, autolag=None
        )[1]
        if heart_stat_p < alpha:
            heart_stat = True
        else:
            heart_stat = False
        eeg_amp_o1_stat_p = adfuller(
            df_all["eeg_amp_o1"].dropna(), regression="ct", maxlag=order, autolag=None
        )[1]
        if eeg_amp_o1_stat_p < alpha:
            eeg_amp_o1_stat = True
        else:
            eeg_amp_o1_stat = False
        if breath_stat and heart_stat and eeg_amp_o1_stat:
            GG = G(df=df_all, max_lag_order=order)
            # 10. write to G_df dataframe
            for mm in GG.index:
                G_df.loc[stage, mm].append([GG[mm], duration])
            textstr = (
                f"Data is stationary: breath - {breath_stat} | "
                f"heart - {heart_stat} | "
                f"eeg - {eeg_amp_o1_stat}"
            )
            stationaritydic[stage][order] += duration
            return None
        else:
            if order > 3:
                checkStatCalcG(df_all, order - 1, duration)
            else:
                if (
                    sleep_stage_series[ll][2] - sleep_stage_series[ll][1]
                    > 10 * timescales[kk]
                ):
                    sleep_stage_series.append(
                        [
                            sleep_stage_series[ll][0],
                            sleep_stage_series[ll][1],
                            sleep_stage_series[ll][1]
                            + (sleep_stage_series[ll][2] - sleep_stage_series[ll][1])
                            / 2,
                        ]
                    )
                    sleep_stage_series.append(
                        [
                            sleep_stage_series[ll][0],
                            sleep_stage_series[ll][1]
                            + (sleep_stage_series[ll][2] - sleep_stage_series[ll][1])
                            / 2,
                            sleep_stage_series[ll][2],
                        ]
                    )
                    return None
                else:
                    stationaritydic[stage][0] += duration
                    return None
    except ValueError:
        # if data is too short, don't use
        breath_stat = False
        heart_stat = False
        eeg_amp_o1_stat = False
        textstr = f"Data too short (n = {len(df_all)}) for stationarity test."
        stationaritydic[stage][0] += duration
        return None


def G(df, max_lag_order):
    """Calculates the strength of Granger causality from causing to caused
    by G = ln(sigma1/sigma2), s. Geweke1982

    Parameters
    ----------
    df : pd.DataFrame
    max_lag_order: int, maximum order for AR model

    Returns
    -------
    G : pd.Series(MultiIndex; level 1: caused, level2: causing; entries:
        G values"""
    coltuples = [
        ("breath", "heart"),
        ("breath", "eeg_amp_o1"),
        ("heart", "breath"),
        ("heart", "eeg_amp_o1"),
        ("eeg_amp_o1", "breath"),
        ("eeg_amp_o1", "heart"),
    ]
    multicol = pd.MultiIndex.from_tuples(coltuples, names=("caused", "causing"))
    res_Series = pd.Series(index=multicol, dtype="object")
    # fit model including all time series
    model_all = VAR(df_all)
    for caused in signals:
        for causing in signals:
            if caused != causing:
                max_lag_order = max_lag_order
                results_all = model_all.fit(maxlags=max_lag_order, ic=None, trend="ct")
                sigma_u_all = results_all.forecast_cov(1)[0]
                names_all = results_all.names
                sigma2 = sigma_u_all[names_all.index(caused)][names_all.index(caused)]
                # fit model that excludes the causing variable
                model_restr = VAR(df_all.drop(columns=causing))
                results_restr = model_restr.fit(
                    maxlags=max_lag_order, ic=None, verbose=True, trend="ct"
                )
                names_restr = results_restr.names
                sigma_u_restr = results_restr.forecast_cov(1)[0]
                sigma1 = sigma_u_restr[names_restr.index(caused)][
                    names_restr.index(caused)
                ]
                if sigma1 > 0 and sigma2 > 0:
                    res_Series.loc[caused, causing] = np.log(sigma1 / sigma2)
                else:
                    res_Series.loc[caused, causing] = 0
                    print("sigma1 < 0 or sigma2 < 0")
    return res_Series


# %% actual algorithm
# 0. Make list of all experiments
subjectsfile = os.path.join("data", f"list_subjects_{group}.txt")

with open(subjectsfile, "r") as patientfile_r:
    patientdata = patientfile_r.readlines()
for ii in range(len(patientdata)):
    patientdata[ii] = patientdata[ii].strip("\n")
    patientdata[ii] = patientdata[ii].split()
# lines 0 and 1 are header
explist = [
    patientdata[ii][0]
    for ii in range(2, len(patientdata))
    if patientdata[ii][1] in ("Airflow", "Chest", "Abdomen")
]

# create MultiIndex DataFrame for G values
# column levels: caused, causing
# index levels: timescale, sleepstage
coltuples = [
    ("breath", "heart"),
    ("breath", "eeg_amp_o1"),
    ("heart", "breath"),
    ("heart", "eeg_amp_o1"),
    ("eeg_amp_o1", "breath"),
    ("eeg_amp_o1", "heart"),
]
multicol = pd.MultiIndex.from_tuples(coltuples, names=("caused", "causing"))
G_df = pd.DataFrame(index=sleepstages, columns=multicol)
global error_df
# error_df: multiindex: (timescale, expname),
# single columns: duration, FitError, InfError
error_multiindex = pd.MultiIndex(
    levels=[[], []], codes=[[], []], names=["timescale", "expname"]
)
error_df = pd.DataFrame(
    index=error_multiindex, columns=["duration", "FitError", "InfError"]
)
stat_df_new = pd.DataFrame(index=sleepstages, columns=timescales)
# for every time scale (i.e. 1 second, 2 seconds, ...)
for kk in range(len(timescales)):
    awakedic = [0 for ii in range(model_lag_order + 1)]
    LSdic = [0 for ii in range(model_lag_order + 1)]
    DSdic = [0 for ii in range(model_lag_order + 1)]
    REMdic = [0 for ii in range(model_lag_order + 1)]
    stationaritydic = {"awake": awakedic, "LS": LSdic, "DS": DSdic, "REM": REMdic}
    # initialize G_df to empty lists
    for row in G_df.index:
        for col in G_df.columns:
            G_df.loc[row, col] = []
    # time measurement
    time_kk = time.time()
    print(f"timescale: {timescales[kk]} seconds")
    print("---------------------------------------------")
    # for each of the 36 files
    for ii in range(len(explist)):
        time_ii = time.time()
        expname = explist[ii]
        # 1. read sleep stages and create sleep_stage_series (list with
        # beginnings and ends of all continuous sleep stages in the
        # experiment)
        with open(
            os.path.join(
                "data",
                "sleep_stages_converted",
                f"{expname}_sleepstages_conv.dat",
            )
        ) as sleepstagedata_r:
            sleepstagedata = sleepstagedata_r.readlines()
        stage_times = [
            int(sleepstagedata[jj].split()[0]) for jj in range(len(sleepstagedata))
        ]
        sleep_stages = [
            sleepstagedata[jj].split()[2] for jj in range(len(sleepstagedata))
        ]
        # create list of all sleep stages and their durations
        sleep_stage_series = []
        start_stage = 0
        for jj in range(1, len(sleep_stages)):
            # start a new sleep stage episode if the current sleep stage is different
            # from the last one of if we arrive at the end of the file
            if sleep_stages[jj] != sleep_stages[jj - 1] or jj == len(sleep_stages):
                sleep_stage_series.append(
                    [sleep_stages[jj - 1], start_stage, stage_times[jj - 1]]
                )
                start_stage = stage_times[jj]
        time_series_df = pd.read_csv(
            os.path.join(
                "csv",
                "timeseries_resampled",
                surmarker_f,
                group,
                f"timeseries_resampled_1hz_{expname}.csv",
            ),
            index_col=0,
        )
        timeline = time_series_df.index
        breath_rate_values = time_series_df["breath"]
        heart_rate_values = time_series_df["heart"]
        eeg_amp_o1 = time_series_df["alpha"]
        # 2. resample time series to new time scale
        # -----------------------------------------
        timeline_kk = []
        breath_rate_values_kk = []
        heart_rate_values_kk = []
        eeg_amp_o1_kk = []
        for mm in range(0, len(timeline), timescales[kk]):
            timeline_kk.append(np.mean(timeline[mm : mm + timescales[kk]]))
            breath_rate_values_kk.append(
                np.mean(breath_rate_values[mm : mm + timescales[kk]])
            )
            heart_rate_values_kk.append(
                np.mean(heart_rate_values[mm : mm + timescales[kk]])
            )
            eeg_amp_o1_kk.append(np.mean(eeg_amp_o1[mm : mm + timescales[kk]]))
        # 3. calculate indices start_index and stop_index of new timeline
        # at which every sleep stage starts and ends

        # for every sleep stage:
        ll = 0
        while ll < len(sleep_stage_series):
            duration = sleep_stage_series[ll][2] - sleep_stage_series[ll][1]
            # stage is one out of {awake, DS, LS, REM}
            stage = sleep_stage_series[ll][0]
            # find out at which index in timeline_kk the sleep stage
            # starts and ends
            # make sure the timeline starts already when sleep phase
            # begins and make sure it's at least 3 times as long as the
            # time scale and make sure the records last long enough for
            # the rare cases where sleep stages were scored longer than
            # the shortest other signal and that the sleep stage is not
            # 'NA' (i.e. has actually been determined)
            if (
                (sleep_stage_series[ll][1] > timeline_kk[0])
                and (
                    sleep_stage_series[ll][2] - sleep_stage_series[ll][1]
                    > 5 * timescales[kk]
                )
                and (sleep_stage_series[ll][2] <= timeline_kk[-1])
                and (sleep_stage_series[ll][0] != "NA")
            ):
                mm = 0
                while timeline_kk[mm] < sleep_stage_series[ll][1]:
                    mm += 1
                start_index = mm
                while timeline_kk[mm] < sleep_stage_series[ll][2]:
                    mm += 1
                stop_index = mm
                # 4. cut patches to include only the window of the
                # sleep stage in question
                df_all = pd.DataFrame(
                    index=range(len(timeline_kk[start_index:stop_index]))
                )
                df_all["breath"] = breath_rate_values_kk[start_index:stop_index]
                df_all["heart"] = heart_rate_values_kk[start_index:stop_index]
                df_all["eeg_amp_o1"] = eeg_amp_o1_kk[start_index:stop_index]

                # 5. normalize time series
                # (subtract mean, divide by std)
                df_all["breath"] -= df_all["breath"].mean()
                df_all["breath"] /= df_all["breath"].std()
                df_all["heart"] -= df_all["heart"].mean()
                df_all["heart"] /= df_all["heart"].std()
                df_all["eeg_amp_o1"] -= df_all["eeg_amp_o1"].mean()
                df_all["eeg_amp_o1"] /= df_all["eeg_amp_o1"].std()

                # 6. calculate G causality using G function (only if all
                # 3 signals (breath, heart, eeg_amp_o1 are stationary;
                # if data too short for stationarity test, don't use it),
                # this is where everything interesting happens
                checkStatCalcG(df_all, model_lag_order, duration)
                # print AIC examples
                # select_model_order(df_all, timescales[kk], stage, ll, duration)
            ll += 1
        # divide number of stationary time by total time
        for sleep_stage in ["awake", "LS", "DS", "REM"]:
            # catch ZeroDivisionError by just setting to zero when there's no data
            if sum(stationaritydic[sleep_stage]) == 0:
                stat_df_new.loc[sleep_stage, timescales[kk]] = 0
            else:
                stat_df_new.loc[sleep_stage, timescales[kk]] = sum(
                    stationaritydic[sleep_stage][1:]
                ) / sum(stationaritydic[sleep_stage])
        print(
            f"time elapsed for timescale: {timescales[kk]} s, # {ii+1} / "
            f"{len(explist)}: {datetime.timedelta(seconds=time.time() - time_ii)}"
        )
    # 7. save DataFrame
    G_df.to_csv(
        os.path.join(
            "csv",
            "G_df",
            f"G_df_{surmarker_n}{group}_lo{model_lag_order}_ts{timescales[kk]}s.csv",
        )
    )
# %% 8. save the stationarity dic
stat_df_new.to_csv(
    os.path.join(
        "csv", f"stat_df_new_{surmarker_n}{group}_lo{model_lag_order}_{band}.csv"
    )
)
