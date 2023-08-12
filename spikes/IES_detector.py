import numpy as np
from get_iEEG_data import *
from IES_helper_functions import *  # _eegfilt,_FindPeaks,_make_fake,_multi_channel_requirement,_car,_electrode_selection


def ies_detector(data, fs, **kwargs):
    """
    Parameters
    data:           np.NDArray - iEEG recordings (m samples x n channels)
    fs:             int - sampling frequency

    kwargs
    tmul:           float - 19 - threshold multiplier
    absthresh:      float - 100 - absolute threshold for spikes
    sur_time:       float - 0.5 - time surrounding a spike to analyze in seconds
    close_to_edge:  float - 0.05 - beginning and ending buffer in seconds
    too_high_abs:   float - threshold for artifact rejection
    spkdur:         Iterable - min and max spike duration thresholds in ms (min, max)
    lpf1:           float - low pass filter cutoff frequency
    hpf:            float - high pass filter cutoff frequency
    fig_path:       String - file path for saving figures

    Returns
    gdf:            np.NDArray - spike locations (m spikes x (peak index, channel))

    """

    def _check(key, var, keyset):
        """
        Internal Function
        Parameters
        key: string or char - key word argument
        var: any - default value for parameter
        keyset: set of user defined kwargs

        Returns
        var if user defined, else default value
        """
        if key in keyset:
            return kwargs[key]
        return var

    keyset = kwargs.keys()

    ### Assigning KWARGS ###############
    tmul = _check("tmul", 19, keyset)  # 25
    absthresh = _check("absthresh", 100, keyset)
    sur_time = _check("sur_time", 0.5, keyset)
    close_to_edge = _check("close_to_edge", 0.05, keyset)
    too_high_abs = _check("too_high_abs", 1e3, keyset)
    # tmul above which I reject it as artifact
    spkdur = _check(
        "spkdur", np.array([15, 260]), keyset
    )  # spike duration must be less than this in ms. It gets converted to points here
    lpf1 = _check("lpf1", 30, keyset)  # low pass filter for spikey component
    hpf = _check("hpf", 7, keyset)  # high pass filter for spikey component
    fig_path = _check("fig_path", "", keyset)
    labels = _check("labels", [], keyset)
    ###################################

    # Assertions and assignments
    if not isinstance(spkdur, np.ndarray):
        spkdur = np.array(spkdur)

    # Receiver and constant variable initialization
    # all_spikes = np.ndarray((1,2),dtype=float)
    all_spikes = []
    nchs = data.shape[1]
    high_passes = np.empty_like(data)
    low_passes = np.empty_like(data)
    spkdur = spkdur * fs / 1000  # change to samples
    labels = np.array(labels)

    for j in range(nchs):  # Loop through each channel and count spikes
        out = []  # initialize preliminary spike receiver
        signal = data[:, j]  # collect channel

        if sum(np.isnan(signal)) > 0:
            continue  # if there are any nans in the signal skip the channel (worth investigating)

        # re-adjust the mean of the signal to be zero
        signal = signal - np.mean(signal)

        # receiver initializeation
        spike_times = []
        spike_durs = []
        spike_amps = []

        # low pass filter to remove artifact
        lpsignal = eegfilt(signal, lpf1, "lowpass", fs)
        # low pass filter
        low_passes[:, j] = lpsignal
        # high pass filter for the 'spike' component
        hpsignal = eegfilt(lpsignal, hpf, "highpass", fs)
        # high pass filter
        high_passes[:, j] = hpsignal  # collect signals for later plotting

        # defining thresholds
        lthresh = np.median(abs(hpsignal))
        # this algorithm might need to be adjusted
        thresh = lthresh * tmul
        # this is the final threshold we want to impose

        for k in range(
            2
        ):  # loop through the positive and negative spikes (could be accomplished without loop)
            if k == 1:
                ksignal = -hpsignal
            else:
                ksignal = hpsignal

            # apply custom peak finder /IES_helper_functions.py
            spp, spv = FindPeaks(ksignal)  # calculate peaks and troughs
            spp, spv = spp.squeeze(), spv.squeeze()  # reformat

            # find the durations less than or equal to that of a spike
            idx = np.argwhere(np.diff(spp) <= spkdur[1]).squeeze()
            startdx = spp[idx]  # indices for each spike that has a long enough duration
            startdx1 = spp[idx + 1]  # indices for each "next" spike
            ### this excludes the last spike: should consider comparing to location of next trough

            # Loop over peaks
            for i in range(len(startdx)):
                spkmintic = spv[(spv > startdx[i]) & (spv < startdx1[i])]
                # find the valley that is between the two peaks
                if not any(spkmintic):
                    continue
                # If the height from valley to either peak is big enough, it could
                # be a spike
                max_height = max(
                    abs(ksignal[startdx1[i]] - ksignal[spkmintic]),
                    abs(ksignal[startdx[i]] - ksignal[spkmintic]),
                )[0]
                if max_height > thresh:  # see if the peaks are big enough
                    spike_times.append(int(spkmintic))  # add index to the spike list
                    spike_durs.append(
                        (startdx1[i] - startdx[i]) * 1000 / fs
                    )  # add spike duration to list
                    spike_amps.append(max_height)  # add spike amplitude to list

        # Generate spikes matrix
        spikes = np.vstack([spike_times, spike_durs, spike_amps]).T

        # initialize exclusion receivers
        toosmall = []
        toosharp = []
        toobig = []

        # now have all the info we need to decide if this thing is a spike or not.
        for i in range(spikes.shape[0]):  # for each spike
            # re-define baseline to be 2 seconds surrounding
            surround = sur_time
            istart = int(
                max(0, np.around(spikes[i, 0] - surround * fs))
            )  # find -2s index, ensuring not to exceed idx bounds
            iend = int(
                min(len(hpsignal), np.around(spikes[i, 0] + surround * fs + 1))
            )  # find +2s index, ensuring not to exceed idx bounds

            alt_thresh = (
                np.median(abs(hpsignal[istart:iend])) * tmul
            )  # identify threshold within this window

            if (spikes[i, 2] > alt_thresh) & (
                spikes[i, 2] > absthresh
            ):  # both parts together are bigger than thresh: so have some flexibility in relative sizes
                if (
                    spikes[i, 1] > spkdur[0]
                ):  # spike wave cannot be too sharp: then it is either too small or noise
                    if spikes[i, 2] < too_high_abs:
                        out.append(
                            spikes[i, 0]
                        )  # add timestamp of spike to output list
                    else:
                        toobig.append(spikes[i, 0])  # spike is above too_high_abs
                else:
                    toosharp.append(spikes[i, 0])  # spike duration is too short
            else:
                toosmall.append(
                    spikes[i, 0]
                )  # window-relative spike height is too short
        out = np.array(out)

        # Spike Realignment
        if out.any():
            # Re-align spikes to peak of the spikey component
            timeToPeak = np.array([-0.15, 0.15])
            # Only look 150 ms before and 150 ms after the currently defined peak
            fullSurround = np.array([-sur_time, sur_time]) * fs
            idxToPeak = timeToPeak * fs

            for i in range(len(out)):
                currIdx = out[i]
                surround_idx = np.arange(
                    max(0, round(currIdx + fullSurround[0])),
                    min(round(currIdx + fullSurround[1]), len(hpsignal)),
                )
                idxToLook = np.arange(
                    max(0, round(currIdx + idxToPeak[0])),
                    min(round(currIdx + idxToPeak[1]), len(hpsignal)),
                )
                snapshot = hpsignal[idxToLook] - np.median(hpsignal[surround_idx])
                # Look at the high frequency signal (where the mean is substracted already)
                I = np.argmax(abs(snapshot))
                # The peak is the maximum absolute value of this
                out[i] = idxToLook[0] + I
                ### Might need to change this to -2 or -0 # changed from -1 to -0 because we want to add index to earlier index

        # Concatenate the list of spikes to the global spike receiver
        if out.any():
            temp = (
                np.array(
                    [np.expand_dims(out, 1), np.tile([j], (len(out), 1))], dtype=object
                )
                .squeeze()
                .T
            )
            # all_spikes = np.vstack((all_spikes,temp))
            all_spikes.append(temp)
            # change all spikes to a list, append and then vstack all at end

    # Final Post-Processing - sort spikes by time not np.isnan(all_spikes).all():
    if len(all_spikes) == 0:
        return np.array([])
    else:
        all_spikes = np.vstack(all_spikes)
        all_spikes = np.vstack(list({tuple(row) for row in list(all_spikes)}))
        idx = np.argsort(all_spikes[:, 0], axis=0)
        gdf = all_spikes[idx, :]

        # if there are no spikes, just give up

        # print('No spikes detected')

    # if all_spikes.any():
    #     # remove exact copies of spikes
    #     all_spikes = np.vstack(list({tuple(row) for row in list(all_spikes)}))

    #     # sort spikes
    #     idx = np.argsort(all_spikes[:,0],axis=0)
    #     gdf = all_spikes[idx,:]

    ## Remove those too close to beginning and end
    if gdf.shape[0]:
        close_idx = close_to_edge * fs
        gdf = gdf[gdf[:, 0] > close_idx, :]
        gdf = gdf[gdf[:, 0] < data.shape[0] - close_idx, :]

    # removing duplicate spikes with closeness threshold
    if gdf.any():
        # distance between spike times
        gdf_diff = np.diff(gdf, axis=0)
        # time difference is below threshold - thresh not necessary becasue of spike realignment
        mask1 = abs(gdf_diff[:, 0]) < 100e-3 * fs
        # ensuring they're on different channels
        mask2 = gdf_diff[:, 1] == 0
        too_close = np.argwhere(mask1 & mask2) + 1

        # coerce shape of mask to <=2 dimensions
        too_close = too_close.squeeze()
        close_mask = np.ones((gdf.shape[0],), dtype=bool)
        close_mask[too_close] = False
        gdf = gdf[close_mask, :]

    # Check that spike occurs in multiple channels
    if gdf.any() & (nchs > 1):
        gdf = multi_channel_requirement(gdf, nchs, fs)

    return gdf
