from scipy import signal, interpolate
from scipy.signal import hilbert
import numpy as np


def calculate_kuramoto_order_parameter(time_series):
    """
    Calculate the Kuramoto order parameter for a set of time series
    Args:
        time_series (np.array): 2D array where each row is a time series
    Returns:
        np.array: Kuramoto order parameter for each time point
    """
    # Extract the number of time series and the number of time points
    num_series, num_time_points = time_series.shape
    # Apply the Hilbert Transform to get an analytical signal
    analytical_signals = hilbert(time_series)
    assert analytical_signals.shape == time_series.shape
    # Extract the instantaneous phase for each time series using np.angle
    phases = np.angle(analytical_signals)
    assert phases.shape == time_series.shape
    # Compute the Kuramoto order parameter for each time point
    # 1j*1j == -1
    r = np.abs(np.sum(np.exp(1j*phases), axis=0)) / num_series
    return r


def process_eeg_data(ecog_data, sample_rate, notch_freq=50, band_pass_freq=(100, 125)):
    """
    Preprocess EEG data by applying a notch filter and a band-pass filter.

    Parameters:
    ecog_data : numpy.ndarray
        Input EEG data. Each row corresponds to a channel and each column corresponds to a time point.
    sample_rate : float
        The sampling rate (number of samples per second) of the EEG data.
    notch_freq : float, optional
        The frequency to be removed by the notch filter (default is 50Hz).
    band_pass_freq : tuple of float, optional
        The frequency band to be allowed through by the bandpass filter (default is (100, 125)).

    Returns:
    ecog_data_band_pass_filtered : numpy.ndarray
        The preprocessed EEG data after applying notch filter and band-pass filter.

    This function applies a notch filter to remove line noise at a specific frequency (default is 50Hz), 
    then applies a band-pass Butterworth filter to keep frequencies within a specific range (default is 100-125Hz).
    Both filters are applied using a zero-phase method (filtfilt), which does not introduce phase delay.
    """
    # Apply notch filter to eliminate 50Hz noise
    notch_b, notch_a = signal.iirnotch(notch_freq, 30, sample_rate)
    ecog_data_notch_filtered = signal.filtfilt(notch_b, notch_a, ecog_data, axis=0)
    
    # Apply bandpass filter
    band_pass_b, band_pass_a = signal.butter(2, band_pass_freq, btype='band', fs=sample_rate)
    ecog_data_band_pass_filtered = signal.filtfilt(band_pass_b, band_pass_a, ecog_data_notch_filtered, axis=0)
    
    return ecog_data_band_pass_filtered