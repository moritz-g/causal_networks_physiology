# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:31:53 2019

@author: moritz

This file uses the raw data and needs no preprocessing before it, everything
should work on the raw data. It creates time series for breath,
heart and eeg data that are resampled to 1s resolution.
Analysis should be continued with process_sleep_data.py

Currently only the O1 EEG signal is processed.

For each subject the best breathing signal was chosen and is listed in the
list_subjects_{group}.txt. From there, the information is retrieved.

reading
-------
0. read index and sleep stage data
1. read breathing peak data
2. read rr interval data
3. read eeg data

processing
----------
4. reconstruct breathing rate time series from breathing data
5. reconstruct heart rate time series from rr data
6. reconstruct eeg amplitudes from eeg data

7. write to file (moved to the ends of step 4, 5 and 6)
"""

# %% imports, setting
from scipy import interpolate
import math
import os
import numpy as np
import datetime
import pandas as pd
import copy

group = "YC"  # 'YC', 'EC', 'OSA' - Young Control, Elderly Control, Sleep Apnea
Surrogate = True
bands = ("delta", "theta", "alpha")
band_limits_dic = {"delta": (0.5, 2), "theta": (4, 7), "alpha": (7.8, 15.6)}
timescales = [1, 2, 5, 10, 15, 30, 60, 120]

# read information on the subjects and extract which breathing signal to use
subjectsfile = os.path.join("data", f"list_subjects_{group}.txt")
with open(subjectsfile, "r") as patientfile_r:
    patientdata = patientfile_r.readlines()
for ii in range(len(patientdata)):
    patientdata[ii] = patientdata[ii].strip("\n")
    patientdata[ii] = patientdata[ii].split()
patientdata = patientdata[2:]

# %% actual routine

for ii in range(len(patientdata)):
    # expname1 is for index data and EEG, expname2 for heart, expname3 for resp
    if Surrogate:
        # pick three different subjects for surrogate mode
        expname1 = patientdata[ii][0]
        expname2 = patientdata[ii - 1][0]
        expname3 = patientdata[ii - 2][0]
    else:
        # pick always the same subject for normal mode
        expname1 = patientdata[ii][0]
        expname2 = patientdata[ii][0]
        expname3 = patientdata[ii][0]
    # reading
    # *******

    # 0. read index and sleep stage data
    # ----------------------------------
    with open(os.path.join("data", "index", f"{expname1}.index"), "r") as indexdata_r:
        indexdata = indexdata_r.readlines()
    # data_redcord_duration indicates how long every data block is [s]
    # this is relevant for the eegdata in order to calculate the sampling rate
    data_record_duration = float(indexdata[8].split(":")[1].strip())
    try:
        with open(
            os.path.join(
                "data",
                "sleep_stages_converted",
                f"{expname1}_sleepstages_conv.dat",
            )
        ) as sleepstagedata_r:
            sleepstagedata = sleepstagedata_r.readlines()
    except FileNotFoundError:
        print("File not found")
        continue
    total_duration_phases = datetime.timedelta(
        seconds=int(sleepstagedata[-1].split()[0])
    )

    # 1. read breathing peak data
    # ---------------------------
    if Surrogate:
        breathmethod = patientdata[ii - 2][1]
    else:
        breathmethod = patientdata[ii][1]
    # skip patients that have no reliable signals, as indicated in the
    # patientdata
    if breathmethod not in ("Airflow", "Chest", "Abdomen"):
        continue
    with open(
        os.path.join(
            "data",
            "breathing",
            "peaks",
            f"{expname3}.{breathmethod}_extrema.dat",
        )
    ) as breathdata_peak_r:
        breathdata_peak = breathdata_peak_r.readlines()
    for jj in range(len(breathdata_peak)):
        breathdata_peak[jj] = breathdata_peak[jj].strip("\n")
        breathdata_peak[jj] = breathdata_peak[jj].split()
    breath_time_max = [
        float(breathdata_peak[jj][0]) for jj in range(len(breathdata_peak))
    ]
    breath_peak_max = [
        float(breathdata_peak[jj][2]) for jj in range(len(breathdata_peak))
    ]
    breath_time_min = [
        float(breathdata_peak[jj][1]) for jj in range(len(breathdata_peak))
    ]
    breath_peak_min = [
        float(breathdata_peak[jj][3]) for jj in range(len(breathdata_peak))
    ]
    total_duration_breathing = datetime.timedelta(seconds=breath_time_min[-1])

    # 2. read rr interval data
    # ------------------------
    with open(
        os.path.join("data", "heartrates", f"{expname2}.rri"), "r"
    ) as heartratedata_r:
        heartratedata = heartratedata_r.readlines()
    # cut header
    header = heartratedata[:5]
    heartratedata = heartratedata[7:]
    # calculate total duration of experiment
    total_duration_heart = 0
    for jj in range(len(heartratedata)):
        heartratedata[jj] = float(heartratedata[jj].strip("\n").split()[0]) / 1000
        total_duration_heart += heartratedata[jj]
    total_duration_heart = datetime.timedelta(seconds=total_duration_heart)

    # 3. read eeg data
    # ----------------
    eegfolder = os.path.join("eegs", group)
    with open(
        os.path.join("data", "eegs", group, f"{expname1}.O1-M2.EEG"), "r"
    ) as eegdata_o1_rr:
        eegdata_o1_r = eegdata_o1_rr.readlines()
    header_o1 = eegdata_o1_r[:9]
    eegdata_o1 = [float(eegdata_o1_r[jj]) for jj in range(11, len(eegdata_o1_r))]
    # calculate sampling rates of eeg files
    record_samples_o1 = int(header_o1[8].split(":")[1].strip())
    f_s_o1 = int(round(record_samples_o1 / data_record_duration, 0))
    total_duration_eeg_o1 = datetime.timedelta(seconds=len(eegdata_o1) / f_s_o1)

    # processing
    # **********

    # print total durations of experiment according to different files
    print("### {} / {}: {}".format(ii + 1, len(patientdata), expname1))
    # total duration of experiment is the minimum of all calculated durations
    min_duration = min(
        total_duration_phases,
        total_duration_breathing,
        total_duration_heart,
        total_duration_eeg_o1,
    )
    max_duration = max(
        total_duration_phases,
        total_duration_breathing,
        total_duration_heart,
        total_duration_eeg_o1,
    )
    max_difference = max_duration - min_duration
    timeline = [jj for jj in range(int(round(min_duration.total_seconds(), 0)))]
    # create skeleton dataframe to put final results in later
    time_series_df = pd.DataFrame(index=timeline)

    # 4. reconstruct breath rate time series
    # --------------------------------------
    breath_rate_times_r = []
    breath_rate_values_r = []
    for jj in range(len(breath_peak_max) - 1):
        breath_rate_times_r.append((breath_time_max[jj + 1] + breath_time_max[jj]) / 2)
        breath_rate_values_r.append(breath_time_max[jj + 1] - breath_time_max[jj])
    breath_rate_interpol = interpolate.interp1d(
        breath_rate_times_r,
        breath_rate_values_r,
        assume_sorted=True,
        bounds_error=False,
    )
    breath_rate_values = breath_rate_interpol(timeline)
    # invert in order to get rate out of the interbreath intervals
    breath_rate_values = [
        1 / breath_rate_values[jj] for jj in range(len(breath_rate_values))
    ]
    # write to final output dataframe
    time_series_df["breath"] = breath_rate_values

    # 5. reconstruct heart rate time series
    # -------------------------------------
    heart_rate_times_r = []
    heart_rate_values_r = heartratedata[1:]
    heart_time_total = heartratedata[0]
    # for times: add up intervals, for values: just take the interval value,
    # that is directly given by the data
    for jj in range(1, len(heartratedata)):
        heart_rate_times_r.append(heart_time_total + heartratedata[jj] / 2)
        heart_time_total += heartratedata[jj]
    heart_rate_interpol = interpolate.interp1d(
        heart_rate_times_r, heart_rate_values_r, assume_sorted=True, bounds_error=False
    )
    heart_rate_values = heart_rate_interpol(timeline)
    # invert in order to get rate out of the heartbeat intervals
    heart_rate_values = [
        1 / heart_rate_values[jj] for jj in range(len(heart_rate_values))
    ]
    # write to final output dataframe
    time_series_df["heart"] = heart_rate_values

    # 6. reconstruct EEG amplitude time series
    # ----------------------------------------
    eeg_times_o1 = [jj / f_s_o1 for jj in range(len(eegdata_o1))]
    #    eeg_times_o2 = [jj/f_s_o2 for jj in range(len(eegdata_o2))]
    # fourier, hilbert and inverse fourier transform incl. band pass
    eegdata_o1_f_o = np.fft.rfft(eegdata_o1)  # to copy from for every band
    fourierfreqs = np.fft.rfftfreq(len(eegdata_o1), d=1 / f_s_o1)
    for band in bands:
        eegdata_o1_f = copy.deepcopy(eegdata_o1_f_o)
        band_limits = band_limits_dic[band]
        for jj in range(len(fourierfreqs)):
            if fourierfreqs[jj] < band_limits[0] or fourierfreqs[jj] > band_limits[1]:
                eegdata_o1_f[jj] = 0
        eegdata_o1_fh = [-1j * eegdata_o1_f[jj] for jj in range(len(eegdata_o1_f))]
        eegdata_o1_h = np.fft.irfft(eegdata_o1_fh)
        eegdata_o1_i = np.fft.irfft(eegdata_o1_f)
        eegdata_o1_compl = [
            eegdata_o1_i[jj] + 1j * eegdata_o1_h[jj] for jj in range(len(eegdata_o1))
        ]
        eeg_amp_o1_r = [
            abs(eegdata_o1_compl[jj]) for jj in range(len(eegdata_o1_compl))
        ]
        # resample to one second by averaging over one second
        eeg_amp_o1 = [
            math.nan,
        ]
        for jj in range(timeline[1], timeline[-1] + 1):
            eeg_amp_o1.append(
                np.mean(
                    eeg_amp_o1_r[
                        int((jj - 1 / 2) * f_s_o1) : int((jj + 1 / 2) * f_s_o1)
                    ]
                )
            )
        # write to final output dataframe
        time_series_df[band] = eeg_amp_o1
    # now breathing, heart and eeg data are time series with 1 Hz sampling
    # rate: breath_rate_values, heart_rate_values. eeg_amp_o1, eeg_amp_o2

    time_series_df.dropna(axis="index", inplace=True)
    if not Surrogate:
        time_series_df.to_csv(
            os.path.join(
                "csv",
                "timeseries_resampled",
                group,
                f"timeseries_resampled_1hz_{expname1}.csv",
            ),
            index_label="time",
            columns=("breath", "heart", "delta", "theta", "alpha"),
        )
    if Surrogate:
        time_series_df.to_csv(
            os.path.join(
                "csv",
                "timeseries_resampled",
                "surrogate",
                group,
                f"timeseries_resampled_1hz_{expname1}.csv",
            ),
            index_label="time",
            columns=("breath", "heart", "delta", "theta", "alpha"),
        )
