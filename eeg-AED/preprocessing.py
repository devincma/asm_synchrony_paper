import pandas as pd
import numpy as np
import scipy as sc
from bipolar_reference import bipolar_reference


def detect_bad_channels(data, fs, channel_labels):
    """
    data: raw EEG traces after filtering (i think)
    fs: sampling frequency
    channel_labels: string labels of channels to use
    """
    values = data.copy()
    which_chs = np.arange(values.shape[1])
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


def reference(data, type, channel_labels):
    if type == "car":
        signal_ref = data - np.expand_dims(np.mean(data, axis=1), 1)
    elif type == "bipolar":
        assert isinstance(channel_labels[0], str)
        dataframe = pd.DataFrame(data, columns=channel_labels)
        signal_df = bipolar_reference(dataframe)
        channel_labels = np.array(signal_df.columns)
        signal_ref = signal_df.to_numpy()
    else:
        raise ("Unrecognized Montage Type")
    return signal_ref, channel_labels


def preprocess(signal, fs, channel_labels, manual_labels, ref_type):
    def ffunc(signal):
        # Should add artifact and nan-removal
        # remove 60Hz noise
        f0 = 60.0  # Frequency to be removed from signal (Hz)
        Q = 30.0  # Quality factor
        b, a = sc.signal.iirnotch(f0, Q, fs)
        signal_filt = sc.signal.filtfilt(b, a, signal, axis=0)

        # bandpass between 1 and 120Hz
        bandpass_b, bandpass_a = sc.signal.butter(3, [1, 120], btype="bandpass", fs=fs)
        signal_filt = sc.signal.filtfilt(bandpass_b, bandpass_a, signal_filt, axis=0)
        return signal_filt

    # excepting if only passing through one channel
    if signal.shape[1] > 1:
        # Check to see if user manually defined channels to use
        if manual_labels:
            labels = channel_labels
            signal_filt = ffunc(signal)
            signal_ref, labels = reference(signal_filt, ref_type, labels)
        else:
            channel_mask, _ = detect_bad_channels(signal, fs, channel_labels)
            signal_mask = signal[:, channel_mask]
            signal_filt = ffunc(signal_mask)
            labels = channel_labels[channel_mask]
            signal_ref, labels = reference(signal_filt, ref_type, labels)

    else:
        signal_filt = ffunc(signal)
        signal_ref = signal_filt
        labels = channel_labels
    return signal_ref, labels
