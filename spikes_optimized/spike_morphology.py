import numpy as np
from scipy import signal


def line_length(x):
    return np.sum(np.absolute(np.ediff1d(x)))


def zero_crossing_around_mean(x):
    x_bar = np.mean(x)
    x_prime = x - x_bar
    return np.sum((x_prime[:-1] * x_prime[1:]) < 0)


def extract_spike_morphology(spike_signal):
    """
    function to find the morphological features of a spike
    major assumption - that the peak is closest to the spike detection

    input: myspike - the single spike to be analyzed
    """
    if len(spike_signal) == 0:
        return None, None, False, "Short segment"

    # detrend the spike (make the mean = 0)
    spike_signal = spike_signal - np.mean(spike_signal)

    # find the peak closest to the spike detection (this will be our main reference point)

    allmaxima = signal.argrelextrema(spike_signal, np.greater)[
        0
    ]  # find all the maximas
    allminima = signal.argrelextrema(spike_signal, np.less)[0]  # find all the minimas

    stndev = np.std(spike_signal)
    peaks_pos = peaks_pos = signal.find_peaks(
        spike_signal[1000 - 50 : 1000 + 50], height=stndev
    )[0]
    peaks_neg = signal.find_peaks(
        -1 * spike_signal[1000 - 50 : 1000 + 50], height=stndev
    )[0]
    peaks_pos = peaks_pos + 950
    peaks_neg = peaks_neg + 950
    combined_peaks = [peaks_pos, peaks_neg]
    combined_peaks = [x for x in combined_peaks for x in x]

    for peaks in combined_peaks:
        if (spike_signal[peaks] > spike_signal[peaks - 3]) & (
            spike_signal[peaks] < spike_signal[peaks + 3]
        ):
            combined_peaks.remove(peaks)
        if (spike_signal[peaks] < spike_signal[peaks - 3]) & (
            spike_signal[peaks] > spike_signal[peaks + 3]
        ):
            combined_peaks.remove(peaks)

    if not combined_peaks:
        peak = None
        left_point = None
        right_point = None
        slow_end = None
        slow_max = None

    else:
        if np.size(combined_peaks) > 1:
            peak_from_mid = [x - 1000 for x in combined_peaks]
            peak_idx = np.argmin(np.abs(peak_from_mid))
            peak = combined_peaks[peak_idx]
        else:
            peak_idx = np.argmax(np.abs(spike_signal[combined_peaks]))
            peak = combined_peaks[peak_idx]

        # find the left and right points

        # from peak we will navigate to either baseline or the next minima/maxima
        # here we will trim down the potential left/right peaks/troughs to the 5 closest to the peak
        if (spike_signal[peak + 3] > spike_signal[peak]) & (
            spike_signal[peak - 3] > spike_signal[peak]
        ):  # negative peak
            left_points_trim = allmaxima[allmaxima < peak][-3::]
            right_points_trim = allmaxima[allmaxima > peak][0:3]
        if (spike_signal[peak + 3] < spike_signal[peak]) & (
            spike_signal[peak - 3] < spike_signal[peak]
        ):  # positive peak
            left_points_trim = allminima[allminima < peak][-3::]
            right_points_trim = allminima[allminima > peak][0:3]

        left_points_trim2 = []
        right_points_trim2 = []
        for i, (left, right) in enumerate(zip(left_points_trim, right_points_trim)):
            if (spike_signal[peak + 3] > spike_signal[peak]) & (
                spike_signal[peak - 3] > spike_signal[peak]
            ):  # negative peak
                if spike_signal[left] > 0.5 * spike_signal[peak]:
                    left_points_trim2.append(left)
                if spike_signal[right] > 0.5 * spike_signal[peak]:
                    right_points_trim2.append(right)
            if (spike_signal[peak + 3] < spike_signal[peak]) & (
                spike_signal[peak - 3] < spike_signal[peak]
            ):  # positive peak
                if spike_signal[left] < 0.5 * spike_signal[peak]:
                    left_points_trim2.append(left)
                if spike_signal[right] < 0.5 * spike_signal[peak]:
                    right_points_trim2.append(right)

        if not left_points_trim2:
            left_points_trim2 = [x for x in left_points_trim]
        if not right_points_trim2:
            right_points_trim2 = [x for x in right_points_trim]

        # find the closest spike with the greatest amplitude difference? try to balance this?
        left_point = []
        right_point = []
        if (spike_signal[peak + 3] > spike_signal[peak]) & (
            spike_signal[peak - 3] > spike_signal[peak]
        ):  # negative peak
            dist_from_peak_left = left_points_trim2 - peak
            dist_from_peak_right = right_points_trim2 - peak
            # restrict what we are looking at by looking at the cloesest to the peak (50 samples from peak)
            left_points_trim2 = [
                x + peak for x in dist_from_peak_left if (x <= 50) & (x >= -50)
            ]
            right_points_trim2 = [
                x + peak for x in dist_from_peak_right if (x <= 50) & (x >= -50)
            ]

            # backup if it doesn't find any (e.g. wide spike)
            if not left_points_trim2:
                left_points_trim2 = [
                    x + peak for x in dist_from_peak_left if (x <= 100) & (x >= -100)
                ]
                if not left_points_trim2:
                    left_points_trim2 = [x for x in left_points_trim]
            if not right_points_trim2:
                right_points_trim2 = [
                    x + peak for x in dist_from_peak_right if (x <= 100) & (x >= -100)
                ]
                if not right_points_trim2:
                    right_points_trim2 = [x for x in right_points_trim]

            if not left_points_trim2:
                left_point = None
                right_point = None

            if not right_points_trim2:
                right_point = None
                left_point = None

            value_leftpoints = spike_signal[left_points_trim2]
            value_rightpoints = spike_signal[right_points_trim2]
            left_value_oi = np.argmax(value_leftpoints)
            right_value_oi = np.argmax(value_rightpoints)
            left_point = left_points_trim2[left_value_oi]
            right_point = right_points_trim2[right_value_oi]

        if (spike_signal[peak + 3] < spike_signal[peak]) & (
            spike_signal[peak - 3] < spike_signal[peak]
        ):  # positive peak
            dist_from_peak_left = left_points_trim2 - peak
            dist_from_peak_right = right_points_trim2 - peak
            # restrict what we are looking at by looking at the cloesest to the peak (50 samples from peak)
            left_points_trim2 = [
                x + peak for x in dist_from_peak_left if (x <= 50) & (x >= -50)
            ]
            right_points_trim2 = [
                x + peak for x in dist_from_peak_right if (x <= 50) & (x >= -50)
            ]

            # backup if it doesn't find any (e.g. wide spike)
            if not left_points_trim2:
                left_points_trim2 = [
                    x + peak for x in dist_from_peak_left if (x <= 100) & (x >= -100)
                ]
                if not left_points_trim2:
                    left_points_trim2 = [x for x in left_points_trim]
            if not right_points_trim2:
                right_points_trim2 = [
                    x + peak for x in dist_from_peak_right if (x <= 100) & (x >= -100)
                ]
                if not right_points_trim2:
                    right_points_trim2 = [x for x in right_points_trim]

            if not left_points_trim2:
                left_point = None
                right_point = None

            if not right_points_trim2:
                right_point = None
                left_point = None

            else:
                value_leftpoints = spike_signal[left_points_trim2]
                value_rightpoints = spike_signal[right_points_trim2]
                left_value_oi = np.argmin(value_leftpoints)
                right_value_oi = np.argmin(value_rightpoints)
                left_point = left_points_trim2[left_value_oi]
                right_point = right_points_trim2[right_value_oi]

        # now we will look for the start and end of the aftergoing slow wave.
        # for positive peaks
        counter = 0
        if (spike_signal[peak + 3] < spike_signal[peak]) & (
            spike_signal[peak - 3] < spike_signal[peak]
        ):  # positive peak
            right_of_right_peaks = [x for x in allmaxima if x > right_point]
            right_of_right_troughs = [x for x in allminima if x > right_point]
            slow_start = right_point

            slow_end = []
            for peaks, troughs in zip(right_of_right_peaks, right_of_right_troughs):
                if zero_crossing_around_mean(spike_signal[right_point:peaks]) >= 1:
                    counter += 1
                if (counter >= 1) | (np.abs(spike_signal[right_point]) >= 100):
                    if (
                        (spike_signal[troughs] < 0)
                        | (spike_signal[troughs] < spike_signal[right_point])
                    ) & (troughs - right_point >= 50):
                        slow_end = troughs
                        break

        # for negative peaks
        if (spike_signal[peak + 3] > spike_signal[peak]) & (
            spike_signal[peak - 3] > spike_signal[peak]
        ):  # negative peak
            right_of_right_peaks = [x for x in allmaxima if x > right_point]
            right_of_right_troughs = [x for x in allminima if x > right_point]
            slow_start = right_point

            slow_end = []
            for peaks, troughs in zip(right_of_right_peaks, right_of_right_troughs):
                if zero_crossing_around_mean(spike_signal[right_point:peaks]) >= 1:
                    counter += 1
                if (counter >= 1) | (np.abs(spike_signal[right_point]) >= 100):
                    if (
                        (spike_signal[peaks] > 0)
                        | (spike_signal[peaks] > spike_signal[right_point])
                    ) & (peaks - right_point >= 50):
                        slow_end = peaks
                        break

        # find slow wave peak
        if slow_end:
            slow_max_idx = np.argmax(spike_signal[right_point:slow_end]) + right_point
            slow_min_idx = np.argmin(spike_signal[right_point:slow_end]) + right_point
            slow_max = spike_signal[slow_max_idx] - spike_signal[slow_min_idx]

        if not slow_end:
            slow_end = None
            slow_max = None

    if not peak:
        rise_amp = None
        decay_amp = None
        slow_width = None
        slow_amp = None
        rise_slope = None
        decay_slope = None
        average_amp = None
        linelen = None

    elif not slow_end:
        rise_amp = np.abs(spike_signal[peak] - spike_signal[left_point])
        decay_amp = np.abs(spike_signal[peak] - spike_signal[right_point])
        slow_width = None
        slow_amp = None
        rise_slope = (spike_signal[peak] - spike_signal[left_point]) / (
            peak - left_point
        )
        decay_slope = (spike_signal[right_point] - spike_signal[peak]) / (
            right_point - peak
        )
        average_amp = rise_amp + decay_amp / 2
        linelen = None

    else:
        rise_amp = np.abs(spike_signal[peak] - spike_signal[left_point])
        decay_amp = np.abs(spike_signal[peak] - spike_signal[right_point])
        slow_width = slow_end - right_point
        slow_amp = slow_max
        rise_slope = (spike_signal[peak] - spike_signal[left_point]) / (
            peak - left_point
        )
        decay_slope = (spike_signal[right_point] - spike_signal[peak]) / (
            right_point - peak
        )
        average_amp = rise_amp + decay_amp / 2
        linelen = line_length(spike_signal[left_point:slow_end])

    basic_features = peak, left_point, right_point, slow_end, slow_max
    advanced_features = (
        rise_amp,
        decay_amp,
        slow_width,
        slow_amp,
        rise_slope,
        decay_slope,
        average_amp,
        linelen,
    )

    if left_point is None or right_point is None or slow_end is None:
        is_valid = False
        bad_reason = "Bad feature"
    else:
        is_valid = True
        bad_reason = None

    return basic_features, advanced_features, is_valid, bad_reason
