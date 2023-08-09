import numpy as np
import pandas as pd


def common_average_montage(ieeg_data):
    """
    Compute the common average montage for iEEG data.

    Parameters:
    - ieeg_data: pandas DataFrame
        Rows are data points, columns are electrode channels.

    Returns:
    - cam_data: pandas DataFrame
        Data after applying the common average montage.
    """

    # Ensure input is a DataFrame
    if not isinstance(ieeg_data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    # Compute the average across all channels
    avg_signal = ieeg_data.mean(axis=1)

    # Subtract the average signal from each channel
    result = ieeg_data.sub(avg_signal, axis=0)

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


def detect_bad_channels(data, fs, channel_labels):
    """
    data: raw EEG traces after filtering (i think)
    fs: sampling frequency
    channel_labels: string labels of channels to use
    """
    values = data.copy()
    which_chs = np.arange(values.shape[1])
    chLabels = channel_labels
    ## Parameters to reject super high variance
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 5e3

    ## Parameter to reject high 60 Hz
    percent_60_hz = 0.7

    ## Parameter to reject electrodes with much higher std than most electrodes
    mult_std = 10

    bad = []
    high_ch = []
    nan_ch = []
    zero_ch = []
    high_var_ch = []
    noisy_ch = []
    all_std = np.empty((len(which_chs), 1))
    all_std[:] = np.nan
    details = {}

    for i in range(len(which_chs)):
        # print(chLabels[i])

        ich = which_chs[i]
        eeg = values[:, ich]
        bl = np.nanmedian(eeg)

        ## Get channel standard deviation
        all_std[i] = np.nanstd(eeg)

        ## Remove channels with nans in more than half
        if sum(np.isnan(eeg)) > 0.5 * len(eeg):
            bad.append(ich)
            nan_ch.append(ich)
            continue

        ## Remove channels with zeros in more than half
        if sum(eeg == 0) > (0.5 * len(eeg)):
            bad.append(ich)
            zero_ch.append(ich)
            continue

        ## Remove channels with too many above absolute thresh

        if sum(abs(eeg - bl) > abs_thresh) > 10:
            bad.append(ich)
            high_ch.append(ich)
            continue

        ## Remove channels if there are rare cases of super high variance above baseline (disconnection, moving, popping)
        pct = np.percentile(eeg, [100 - tile, tile])
        thresh = [bl - mult * (bl - pct[0]), bl + mult * (pct[1] - bl)]
        sum_outside = sum(((eeg > thresh[1]) + (eeg < thresh[0])) > 0)
        if sum_outside >= num_above:
            bad.append(ich)
            high_var_ch.append(ich)
            continue

        ## Remove channels with a lot of 60 Hz noise, suggesting poor impedance

        # Calculate fft
        # orig_eeg = orig_values(:,ich)
        # Y = fft(orig_eeg-mean(orig_eeg))
        Y = np.fft.fft(eeg - np.nanmean(eeg))

        # Get power
        P = abs(Y) ** 2
        freqs = np.linspace(0, fs, len(P) + 1)
        freqs = freqs[:-1]

        # Take first half
        P = P[: np.ceil(len(P) / 2).astype(int)]
        freqs = freqs[: np.ceil(len(freqs) / 2).astype(int)]

        P_60Hz = sum(P[(freqs > 58) * (freqs < 62)]) / sum(P)
        if P_60Hz > percent_60_hz:
            bad.append(ich)
            noisy_ch.append(ich)
            continue

    ## Remove channels for whom the std is much larger than the baseline
    median_std = np.nanmedian(all_std)
    higher_std = which_chs[(all_std > (mult_std * median_std)).squeeze()]
    bad_std = higher_std
    for ch in bad_std:
        if ch not in bad:
            bad.append(ch)
    channel_mask = [i for i in which_chs if i not in bad]
    details["noisy"] = noisy_ch
    details["nans"] = nan_ch
    details["zeros"] = zero_ch
    details["var"] = high_var_ch
    details["higher_std"] = bad_std
    details["high_voltage"] = high_ch

    return channel_mask, details
