import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch


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


def identify_bad_channels(values, channel_indices, fs):
    """
    Identifies 'bad' channels in an EEG dataset based on various criteria such as high variance, missing data,
    crossing absolute threshold, high variance above baseline, and 60 Hz noise.

    Parameters:
    values (numpy.ndarray): A 2D array of EEG data where each column is a different channel and each row is a reading.
    channel_indices (list): A list containing indices of channels to be analyzed.
    fs (float): The sampling frequency.

    Returns:
    bad (list): A list of 'bad' channel indices.
    details (dict): A dictionary containing the reasons why each channel was marked as 'bad'. Keys are 'noisy', 'nans',
                    'zeros', 'var', 'higher_std', and 'high_voltage'. Each key maps to a list of channel indices.
    """

    # set parameters
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 5e3
    percent_60_hz = 0.99
    mult_std = 10

    bad = []
    high_ch = []
    nan_ch = []
    zero_ch = []
    high_var_ch = []
    noisy_ch = []
    all_std = np.full(len(channel_indices), np.nan)

    for i in range(len(channel_indices)):
        bad_ch = 0
        ich = channel_indices[i]
        eeg = values[:, ich]
        bl = np.nanmedian(eeg)

        all_std[i] = np.nanstd(eeg)

        if np.sum(np.isnan(eeg)) > 0.5 * len(eeg):
            bad.append(ich)
            nan_ch.append(ich)
            continue

        if np.sum(eeg == 0) > 0.5 * len(eeg):
            bad.append(ich)
            zero_ch.append(ich)
            continue

        if np.sum(np.abs(eeg - bl) > abs_thresh) > 10:
            bad.append(ich)
            bad_ch = 1
            high_ch.append(ich)

        if bad_ch == 1:
            continue

        pct = np.percentile(eeg, [100 - tile, tile])
        thresh = [bl - mult * (bl - pct[0]), bl + mult * (pct[1] - bl)]
        sum_outside = np.sum((eeg > thresh[1]) | (eeg < thresh[0]))

        if sum_outside >= num_above:
            bad_ch = 1

        if bad_ch == 1:
            bad.append(ich)
            high_var_ch.append(ich)
            continue

        Y = fft(eeg - np.nanmean(eeg))

        P = np.abs(Y) ** 2
        freqs = np.linspace(0, fs, len(P) + 1)
        freqs = freqs[:-1]
        P = P[: int(np.ceil(len(P) / 2))]
        freqs = freqs[: int(np.ceil(len(freqs) / 2))]

        total_P = np.sum(P)
        if total_P != 0 and not np.isnan(total_P):
            P_60Hz = np.sum(P[(freqs > 58) & (freqs < 62)]) / total_P
        else:
            P_60Hz = 0  # or any other value that makes sense in the context

        if P_60Hz > percent_60_hz:
            bad_ch = 1

        if bad_ch == 1:
            bad.append(ich)
            noisy_ch.append(ich)
            continue

    median_std = np.nanmedian(all_std)
    higher_std = [
        channel_indices[i]
        for i in range(len(all_std))
        if all_std[i] > mult_std * median_std
    ]
    bad_std = [ch for ch in higher_std if ch not in bad]
    bad.extend(bad_std)

    details = {
        "noisy": noisy_ch,
        "nans": nan_ch,
        "zeros": zero_ch,
        "var": high_var_ch,
        "higher_std": bad_std,
        "high_voltage": high_ch,
    }

    return bad, details


# def detect_bad_channels(values, fs, channel_labels):
#     """
#     data: raw EEG traces after filtering (i think)
#     fs: sampling frequency
#     channel_labels: string labels of channels to use
#     """
#     which_chs = np.arange(values.shape[1])
#     chLabels = channel_labels
#     ## Parameters to reject super high variance
#     tile = 99
#     mult = 10
#     num_above = 1
#     abs_thresh = 5e3

#     ## Parameter to reject high 60 Hz
#     percent_60_hz = 0.7

#     ## Parameter to reject electrodes with much higher std than most electrodes
#     mult_std = 10

#     bad = []
#     high_ch = []
#     nan_ch = []
#     zero_ch = []
#     high_var_ch = []
#     noisy_ch = []
#     all_std = np.empty((len(which_chs), 1))
#     all_std[:] = np.nan
#     details = {}

#     for i in range(len(which_chs)):
#         # print(chLabels[i])

#         ich = which_chs[i]
#         eeg = values[:, ich]
#         bl = np.nanmedian(eeg)

#         ## Get channel standard deviation
#         all_std[i] = np.nanstd(eeg)

#         ## Remove channels with nans in more than half
#         if sum(np.isnan(eeg)) > 0.5 * len(eeg):
#             bad.append(ich)
#             nan_ch.append(ich)
#             continue

#         ## Remove channels with zeros in more than half
#         if sum(eeg == 0) > (0.5 * len(eeg)):
#             bad.append(ich)
#             zero_ch.append(ich)
#             continue

#         ## Remove channels with too many above absolute thresh

#         if sum(abs(eeg - bl) > abs_thresh) > 10:
#             bad.append(ich)
#             high_ch.append(ich)
#             continue

#         ## Remove channels if there are rare cases of super high variance above baseline (disconnection, moving, popping)
#         pct = np.percentile(eeg, [100 - tile, tile])
#         thresh = [bl - mult * (bl - pct[0]), bl + mult * (pct[1] - bl)]
#         sum_outside = sum(((eeg > thresh[1]) + (eeg < thresh[0])) > 0)
#         if sum_outside >= num_above:
#             bad.append(ich)
#             high_var_ch.append(ich)
#             continue

#         ## Remove channels with a lot of 60 Hz noise, suggesting poor impedance

#         # Calculate fft
#         # orig_eeg = orig_values(:,ich)
#         # Y = fft(orig_eeg-mean(orig_eeg))
#         Y = np.fft.fft(eeg - np.nanmean(eeg))

#         # Get power
#         P = abs(Y) ** 2
#         freqs = np.linspace(0, fs, len(P) + 1)
#         freqs = freqs[:-1]

#         # Take first half
#         P = P[: np.ceil(len(P) / 2).astype(int)]
#         freqs = freqs[: np.ceil(len(freqs) / 2).astype(int)]

#         P_60Hz = sum(P[(freqs > 58) * (freqs < 62)]) / sum(P)
#         if P_60Hz > percent_60_hz:
#             bad.append(ich)
#             noisy_ch.append(ich)
#             continue

#     ## Remove channels for whom the std is much larger than the baseline
#     median_std = np.nanmedian(all_std)
#     higher_std = which_chs[(all_std > (mult_std * median_std)).squeeze()]
#     bad_std = higher_std
#     for ch in bad_std:
#         if ch not in bad:
#             bad.append(ch)
#     channel_mask = [i for i in which_chs if i not in bad]
#     details["noisy"] = noisy_ch
#     details["nans"] = nan_ch
#     details["zeros"] = zero_ch
#     details["var"] = high_var_ch
#     details["higher_std"] = bad_std
#     details["high_voltage"] = high_ch

#     return channel_mask, details


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
        denominator = np.sum(P)
        if denominator != 0:
            P_60Hz = np.sum(P[(freqs > 58) & (freqs < 62)]) / denominator
        else:
            print(P)
            P_60Hz = 0

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
