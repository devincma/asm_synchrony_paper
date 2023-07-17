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


def process_eeg_data(data, sample_rate, notch_freq=50, band_pass_freq=(100, 125)):
    """
    Applies a notch filter and a bandpass filter to the input data along each row.
    
    The notch filter helps eliminate noise at a specified frequency (default is 50Hz). 
    The bandpass filter only allows frequencies within a specified range to pass through (default is 100-125Hz).

    Parameters:
    data (numpy.ndarray): Input data where each row represents a channel and each column represents a sample.
    sample_rate (float): The sampling rate of the data in Hz.
    notch_freq (float, optional): The frequency to be eliminated by the notch filter in Hz. Defaults to 50Hz.
    band_pass_freq (tuple, optional): The lower and upper frequency limits of the bandpass filter in Hz. Defaults to (100, 125Hz).

    Returns:
    numpy.ndarray: The filtered data, with the same shape as the input data.
    """
    
    # Apply notch filter to eliminate 50Hz noise
    notch_b, notch_a = signal.iirnotch(notch_freq, 30, sample_rate)
    data_notch_filtered = signal.filtfilt(notch_b, notch_a, data, axis=1)
    
    # Apply bandpass filter
    band_pass_b, band_pass_a = signal.butter(2, band_pass_freq, btype='band', fs=sample_rate)
    data_band_pass_filtered = signal.filtfilt(band_pass_b, band_pass_a, data_notch_filtered, axis=1)
    
    return data_band_pass_filtered
