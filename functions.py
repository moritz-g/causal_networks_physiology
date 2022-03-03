#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class definitions. Function definitions.

@author: moritz
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
import math
from statsmodels.tsa.stattools import acf


def moving_average(signal, window_length):
    return np.convolve(signal, np.ones((window_length,)) / window_length, mode="same")


def create_noise(sample_size=2 ** 16, fs=1, exp=1) -> pd.Series:
    """Creates pandas series, which is 1/f**(exp) noise of
    length sample_size with sampling frequency fs (Hertz).

    Creates white noise first, then fourier transforms,
    multiplies fourier coefficients with 1/f, backtransforms, normalizes"""
    # create random signal (rs)
    rs = [rd.uniform(-0.5, 0.5) for x in range(sample_size)]
    # fourier transform and apply low pass filter (lp), backtransform
    rs_fft = np.fft.rfft(rs)
    rs_fft_lp = [
        rs_fft[0],
    ] + ([rs_fft[x] / (math.sqrt(x) ** (exp)) for x in range(1, len(rs_fft))])
    rs_lp = np.fft.irfft(rs_fft_lp)
    rs_lp_norm = [rs_lp[ii] - np.mean(rs_lp) for ii in range(len(rs_lp))]
    # normalize by dividing by standard deviation (std) --> mean 0, sd 1
    std = np.std(rs_lp_norm)
    for ii in range(len(rs_lp_norm)):
        rs_lp_norm[ii] /= std

    return pd.Series(
        rs_lp_norm,
        index=pd.DatetimeIndex(
            data=pd.date_range(start=0, freq=str(fs) + "s", periods=sample_size)
        ),
    )


def create_gaussian_white_noise(sample_size=2 ** 16, mean=0, std=1):
    samples = np.random.normal(mean, std, size=sample_size)
    return samples


def mprsa(signals, s, L: int, crit=None, normalize=False, logical_operation="and"):
    """
    Create multivariate PRSA timeseries for signal s influenced by signals
    under the criteria crit

    Parameters
    ----------
    - signals: list of signals or integer.
      Each signal is one time series that influences
      (or does not influence) the signal s. Used to create anchor points
      if integer, random anchor points are created
    - s: target signal
    - L: int, BPRSA time series will range from -L to L data points,
      so is 2*L data points long
    - crit: list of anchor point creation criteria, each corresponds to a
      signal in signals
      inc_1: create anchor point whenever s1 increases
      inc_x: create anchor point whenever average of x values in s1
      increases (x integer)
      None: create an anchor point at every data point
    - normalize: Bool
    - logical operation: "or" or "and". Determines if all criteria have to be
      met for anchor point creation or just one.

    Returns
    -------
    tuple: (pd.DataFrame{"MPRSA", "Std"}, number of created anchors)



    Reference
    ---------
    - Schumann et al.: 'Bivariate phase-rectified signal averaging'
      (Physica A 287, 2008)
    - 'Phase-rectified signal averaging detects quasi-periodicities
      in non-stationary data' (Physica A 364, 2006)
    """
    # if signals is an integer, create this amount of random anchors
    if isinstance(signals, int):
        anchors = rd.sample(range(L, len(s) - L), signals)
    else:
        # make list of anchor points
        # idea: at first every time is an anchor point, keep only the ones that
        # fulfill the criteria by using set operiations
        anchors_unprocessed = [[] for ii in range(len(signals))]
        # anchor point creation criterion;
        # this leaves space for further, more complicated anchor point criteria
        # average over T samples has to increase
        for ii in range(len(crit)):
            if crit[ii][0:4] == "inc_":
                T = int(crit[ii][4:])
                for jj in range(L + T, len(signals[ii]) - L - T):
                    if sum(signals[ii][jj : jj + T]) > sum(signals[ii][jj - T : jj]):
                        anchors_unprocessed[ii].append(jj)
            elif crit[ii] is None:
                anchors_unprocessed[ii] = range(len(signals[ii]))
            else:
                raise ValueError("invalid 'crit' variable")
        # logically connect the anchor points from the unprocessed list and get
        # to final anchor point list
        if logical_operation == "and":
            # make intersection of anchor point sets
            anchors = set(range(len(s)))
            for ii in range(len(anchors_unprocessed)):
                anchors = anchors.intersection(anchors_unprocessed[ii])
        elif logical_operation == "or":
            # make union of anchor point sets
            anchors = []
            for ii in range(len(anchors_unprocessed)):
                anchors = anchors.union(anchors_unprocessed[ii])

    # create mean MPRSA time series
    mprsa_ts = []
    mprsa_std_ts = []
    anchors = list(anchors)  # because sets don't support indexing
    for ii in range(-L, L):
        ii_list = [s[anchors[jj] + ii] for jj in range(len(anchors))]
        # can this be improved by using iterators?
        mprsa_ts.append(np.mean(ii_list))
        # calculate errorbars
        mprsa_std_ts.append(np.std(ii_list) / math.sqrt(len(anchors)))
    # normalizing procedure: correct by mean and standard deviation of s2
    if normalize:
        print("MPRSA is normalized to 1")
        if s.std() == 0:
            raise ValueError("standard deviation of target signal is zero")
        mprsa_ts = (mprsa_ts - s.mean()) / s.std()
        mprsa_std_ts = mprsa_std_ts / s.std()
    return (
        pd.DataFrame(
            {"MPRSA": mprsa_ts, "STD": mprsa_std_ts},
            index=[-L + ii for ii in range(len(mprsa_ts))],
        ),
        anchors,
    )
