import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig


def eegfilt(signal, fc, filttype, fs):
    assert fc < fs / 2, "Cutoff frequency must be < one half the sampling rate"
    # initialize constants for filter
    order = 6
    # Create and apply filter
    B, A = sig.butter(order, fc, filttype, fs=fs)
    return sig.filtfilt(B, A, signal)


def FindPeaks(signal):
    """
    ## finds peaks and troughs
    ## Nagi Hatoum
    ## copyright 2005
    ## Adapted from Erin Conrad
    """
    ds = np.diff(signal, axis=0)
    ds = np.insert(ds, 0, ds[0])  # pad diff
    mask = np.argwhere(abs(ds[1:]) <= 1e-3).squeeze()  # got rid of +1
    ds[mask] = ds[mask - 1]
    ds = np.sign(ds)
    ds = np.diff(ds)
    ds = np.insert(ds, 0, ds[0])
    t = np.argwhere(ds > 0)
    p = np.argwhere(ds < 0)
    return p, t


def make_fake(n_spikes=3):
    toy = lambda x, a, b: np.sin((x * a * 2 * np.pi - b * (2 * np.pi)))
    tfs = 200
    slen = 10
    x = np.linspace(0, slen, slen * tfs)
    data1 = toy(x, 10, 0) + toy(x, 25, 0)
    data2 = toy(x, 15, 0) + toy(x, 27, 0)
    data = np.array([data1, data2]).T
    np.random.seed(2 * 3 * 5 * 8 * 13)
    data += np.random.normal(size=data.shape)
    data *= 100
    for s in range(n_spikes):
        base = np.random.randint(tfs, data.shape[0] - tfs)
        data[base : base + 20, :] *= 5
    return data


def multi_channel_requirement(gdf, nchs, fs):
    # Need to change so that it returns total spike counter to help remove duplicates
    min_chs = 2
    if nchs < 16:
        max_chs = np.inf
    else:
        max_chs = np.ceil(nchs / 2)
    min_time = 100 * 1e-3 * fs

    # Check if there is even more than 1 spiking channel. Will throw error
    try:
        if len(np.unique(gdf[:, 1])) < min_chs:
            return np.array([])
    except IndexError:
        return np.array([])

    final_spikes = []

    s = 0  # start at time negative one for 0 based indexing
    curr_seq = [s]
    last_time = gdf[s, 0]
    spike_count = 0
    while s < (gdf.shape[0] - 1):  # check to see if we are at last spike
        # move to next spike time
        new_time = gdf[s + 1, 0]  # calculate the next spike time

        # if it's within the time diff
        if (
            new_time - last_time
        ) < min_time:  # check that the spikes are within the window of time
            curr_seq.append(s + 1)  # append it to the current sequence

            if s == (
                gdf.shape[0] - 2
            ):  # see if you just added the last spike, if so, done with sequence
                # done with sequence, check if the number of involved chs is
                # appropriate
                l = len(np.unique(gdf[curr_seq, 1]))
                if l >= min_chs and l <= max_chs:
                    final_spikes.append(
                        np.hstack(
                            (
                                gdf[curr_seq, :],
                                np.ones((len(curr_seq), 1)) * spike_count,
                            )
                        )
                    )
        else:
            # done with sequence, check if the length of sequence is
            # appropriate
            l = len(np.unique(gdf[curr_seq, 1]))
            if (l >= min_chs) & (l <= max_chs):
                final_spikes.append(
                    np.hstack(
                        (gdf[curr_seq, :], np.ones((len(curr_seq), 1)) * spike_count)
                    )
                )
                spike_count += 1
            # reset sequence
            curr_seq = [s + 1]
        # increase the last time
        last_time = gdf[s + 1, 0]

        # increase the current spike
        s += 1

    if len(final_spikes) != 0:
        return np.vstack(
            final_spikes
        )  # return all final spikes with a spike sequence counter
    else:
        return np.array([])


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


def identify_bad_channels(values, channel_indices, channel_labels, fs):
    """
    Identifies 'bad' channels in an EEG dataset based on various criteria such as high variance, missing data,
    crossing absolute threshold, high variance above baseline, and 60 Hz noise.

    Parameters:
    values (numpy.ndarray): A 2D array of EEG data where each column is a different channel and each row is a reading.
    channel_indices (list): A list containing indices of channels to be analyzed.
    channel_labels (list): A list of channel labels.
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


def detect_bad_channels(values, fs, channel_labels):
    """
    data: raw EEG traces after filtering (i think)
    fs: sampling frequency
    channel_labels: string labels of channels to use
    """
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
