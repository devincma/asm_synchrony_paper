from select import select
import numpy as np
from numpy import matlib
import scipy as sc
from scipy import signal as sig
import matplotlib.pyplot as plt
from get_iEEG_data import *
from IES_helper_functions import *  # _eegfilt,_FindPeaks,_make_fake,_multi_channel_requirement,_car,_electrode_selection
import sys


def ies_detector(data=None, fs: int = 200, **kwargs):
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
    plot_sig:       bool - save default figures
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
    tmul = _check("tmul", 19, keyset)
    absthresh = _check("absthresh", 100, keyset)
    sur_time = _check("sur_time", 0.5, keyset)
    close_to_edge = _check("close_to_edge", 0.05, keyset)
    too_high_abs = _check("too_high_abs", 1e3, keyset)
    # tmul above which I reject it as artifact
    spkdur = _check(
        "spkdur", np.array([15, 200]), keyset
    )  # spike duration must be less than this in ms. It gets converted to points here
    lpf1 = _check("lpf1", 30, keyset)  # low pass filter for spikey component
    hpf = _check("hpf", 7, keyset)  # high pass filter for spikey component
    plot_sig = _check("plot_sig", True, keyset)
    fig_path = _check("fig_path", "", keyset)
    labels = _check("labels", [], keyset)
    ###################################

    # Assertions and assignments
    if data is None:
        data = make_fake()
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

        # plotting the different filtered signals
        if plot_sig:
            fig, axs1 = plt.subplots(3, 1)
            axs1[0].plot(signal)
            axs1[0].set_title("Raw Signal")
            axs1[1].plot(lpsignal)
            axs1[1].set_title("Low Pass Signal")
            axs1[2].plot(hpsignal)
            axs1[2].set_title("High Pass Signal")
            fig.savefig(fig_path + "filtering_plot.png")
            plt.close(fig)

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
        if plot_sig and out.any():
            fig, ax = plt.subplots(3, 1)
            ax[0].plot(signal)
            ax[0].plot(out, signal[out.astype(int)], "ro")
            ax[1].plot(lpsignal)
            ax[1].plot(out, lpsignal[out.astype(int)], "ro")
            ax[2].plot(hpsignal)
            ax[2].plot(out, hpsignal[out.astype(int)], "ro")
            fig.savefig(fig_path + "channel_spikes.png")
            plt.close(fig)

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

    # plot the waveforms of all spikes and spike locations on all channels
    if plot_sig:
        pre_spike = 0.025
        post_spike = 0.025
        pre_idxs = pre_spike * fs
        post_idxs = post_spike * fs
        aligned_time = np.arange(-pre_spike, post_spike, 1 / fs)
        fig, axs = plt.subplots()
        if gdf.any():
            _, spidxs = np.unique(gdf[:, 1], return_index=True)
            for spike in range(gdf.shape[0]):
                if spike + post_idxs < len(hpsignal):
                    mask = np.arange(spike - pre_idxs, spike + post_idxs, dtype=int)
                    axs.plot(aligned_time, hpsignal[mask])
        fig.savefig(fig_path + "spike_plot.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(24, 8))  ### CHANGE THIS FOR MORE CHANNELS
        # channels = np.unique(gdf[:,1]).astype(int)
        temp = data  # [:,channels]
        # temp_labels = labels[channels]

        channel_offsets = np.insert(
            np.cumsum(
                np.abs(np.min(temp[:, 1:], axis=0)) + np.max(temp[:, :-1], axis=0)
            ),
            0,
            0,
        )

        temp2 = temp + channel_offsets
        time = np.arange(len(temp2)) / fs
        ax.plot(time, temp2, "k", linewidth=0.2)
        if gdf.any():
            for idx in range(gdf.shape[0]):
                spike = gdf[idx, 0]
                plt.plot(
                    time[int(spike)],
                    temp2[int(gdf[idx, 0]), int(gdf[idx, 1])],
                    "ro",
                    alpha=1,
                )
        ax.set_yticks(channel_offsets)
        ax.set_yticklabels(labels)
        fig.savefig(fig_path + "all_spikes.png")
        plt.close(fig)

    return gdf


def main():
    from ieeg.auth import Session

    with open("/gdrive/public/USERS/wojemann/woj_ieeglogin.bin", "r") as f:
        s = Session(
            "wojemann", f.read()
        )  # start an IEEG session with your username and password. TODO: where should people put their ieeg_pwd.bin file?
    ds = s.open_dataset("HUP212_phaseII")
    all_channel_labels = np.array(ds.get_channel_labels())
    label_idxs = electrode_selection(all_channel_labels)
    labels = all_channel_labels[label_idxs]
    data, fs = get_iEEG_data(
        "wojemann",
        "/gdrive/public/USERS/wojemann/woj_ieeglogin.bin",
        "HUP212_phaseII",
        100000 * 1e6,
        100015 * 1e6,
        labels,
    )  # [4,14,34,39,40,55,67])
    data = car(data)
    signal = data.to_numpy()
    output = ies_detector(
        data=signal,
        fs=fs,
        plot_sig=True,
        labels=labels,
        fig_path="/gdrive/public/USERS/wojemann/iEEG_processing/figures/",
    )
    print(len(np.unique(output[:, 2])), "spikes detected")
    print(output)


if __name__ == "__main__":
    main()
