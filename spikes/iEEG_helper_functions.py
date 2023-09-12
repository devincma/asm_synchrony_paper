# This file should not use pandas

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, hilbert


def calculate_synchrony(time_series):
    """
    Calculate the Kuramoto order parameter for a set of time series
    Args:
        time_series (np.array): 2D array where each row is a time series
    Returns:
        np.array: Kuramoto order parameter for each time point
    """
    # Extract the number of time series and the number of time points
    N, _ = time_series.shape
    # Apply the Hilbert Transform to get an analytical signal
    analytical_signals = hilbert(time_series)
    assert analytical_signals.shape == time_series.shape
    # Extract the instantaneous phase for each time series using np.angle
    phases = np.angle(analytical_signals, deg=False)
    assert phases.shape == time_series.shape
    # Compute the Kuramoto order parameter for each time point
    # 1j*1j == -1
    r_t = np.abs(np.sum(np.exp(1j * phases), axis=0)) / N
    R = np.mean(r_t)
    return r_t, R


def notch_filter(data, low_cut, high_cut, fs, order=4):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = iirnotch(w0=(low + high) / 2, Q=30, fs=fs)
    y = filtfilt(b, a, data, axis=0)
    return y


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data, axis=0)
    return y


def common_average_montage(ieeg_data):
    """
    Compute the common average montage for iEEG data.

    Parameters:
    - ieeg_data: 2D numpy array
        Rows are data points, columns are electrode channels.

    Returns:
    - cam_data: 2D numpy array
        Data after applying the common average montage.
    """

    # Ensure input is a numpy array
    if not isinstance(ieeg_data, np.ndarray):
        raise ValueError("Input data must be a 2D numpy array.")

    # Ensure the shape of ieeg_data is correct
    if ieeg_data.shape[0] < ieeg_data.shape[1]:
        raise ValueError("ieeg_data must have more rows than columns. ")

    # Compute the average across all channels
    avg_signal = ieeg_data.mean(axis=1)

    # Subtract the average signal from each channel
    result = ieeg_data - avg_signal[:, np.newaxis]

    # Check if the shape of the result matches the shape of ieeg_data
    if result.shape != ieeg_data.shape:
        raise ValueError(
            "The shape of the resulting data doesn't match the input data."
        )

    return result


def electrode_selection(labels):
    """
    returns label selection array
    inputs:
    labels - string array of channel label names
    """
    select = np.ones((len(labels),), dtype=bool)
    for i, label in enumerate(labels):
        label = label.upper()
        for check in ["EKG", "ECG", "RATE", "RR"]:
            if check in label:
                select[i] = 0

        checks = set(
            (
                "C3",
                "C4",
                "CZ",
                "F8",
                "F7",
                "F4",
                "F3",
                "FP2",
                "FP1",
                "FZ",
                "LOC",
                "T4",
                "T5",
                "T3",
                "C6",
                "ROC",
                "P4",
                "P3",
                "T6",
            )
        )
        if label in checks:
            select[i] = 0

        # fix for things that could be either scalp or ieeg
        if label == "O2":
            if "O1" in set(
                labels
            ):  # if hemiscalp, should not have odd; if ieeg, should have O1
                select[i] = 0
    return select


def detect_bad_channels_optimized(values, fs):
    which_chs = np.arange(values.shape[1])

    ## Parameters
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 5e3
    percent_60_hz = 0.7
    mult_std = 10

    bad = set()
    high_ch = []
    nan_ch = []
    zero_ch = []
    high_var_ch = []
    noisy_ch = []

    nans_mask = np.isnan(values)
    zero_mask = values == 0
    nan_count = np.sum(nans_mask, axis=0)
    zero_count = np.sum(zero_mask, axis=0)

    median_values = np.nanmedian(values, axis=0)
    std_values = np.nanstd(values, axis=0)

    median_std = np.nanmedian(std_values)
    higher_std = which_chs[std_values > (mult_std * median_std)]

    for ich in which_chs:
        eeg = values[:, ich]

        # Check NaNs
        if nan_count[ich] > 0.5 * len(eeg):
            bad.add(ich)
            nan_ch.append(ich)
            continue

        # Check zeros
        if zero_count[ich] > (0.5 * len(eeg)):
            bad.add(ich)
            zero_ch.append(ich)
            continue

        # Check above absolute threshold
        if np.sum(np.abs(eeg - median_values[ich]) > abs_thresh) > 10:
            bad.add(ich)
            high_ch.append(ich)
            continue

        # High variance check
        pct = np.percentile(eeg, [100 - tile, tile])
        thresh = [
            median_values[ich] - mult * (median_values[ich] - pct[0]),
            median_values[ich] + mult * (pct[1] - median_values[ich]),
        ]
        if np.sum((eeg > thresh[1]) | (eeg < thresh[0])) >= num_above:
            bad.add(ich)
            high_var_ch.append(ich)
            continue

        # 60 Hz noise check, modified to match original function
        Y = np.fft.fft(eeg - np.nanmean(eeg))
        P = np.abs(Y) ** 2
        freqs = np.linspace(0, fs, len(P) + 1)
        freqs = freqs[:-1]
        P = P[: int(np.ceil(len(P) / 2))]
        freqs = freqs[: int(np.ceil(len(freqs) / 2))]
        total_power = np.sum(P)
        if total_power == 0:
            bad.add(ich)
            high_var_ch.append(ich)
            continue
        else:
            P_60Hz = np.sum(P[(freqs > 58) & (freqs < 62)]) / total_power
            if P_60Hz > percent_60_hz:
                bad.add(ich)
                noisy_ch.append(ich)

    # Combine all bad channels
    bad = bad.union(higher_std)

    details = {
        "noisy": noisy_ch,
        "nans": nan_ch,
        "zeros": zero_ch,
        "var": high_var_ch,
        "higher_std": list(higher_std),
        "high_voltage": high_ch,
    }

    channel_mask = [i for i in which_chs if i not in bad]

    return channel_mask, details
