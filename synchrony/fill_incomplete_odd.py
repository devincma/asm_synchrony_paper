#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
from ieeg.auth import Session

from get_iEEG_data import *
from iEEG_helper_functions import *


# In[ ]:


SYNCHRONY_BROADBAND_DIRECTORY = "../../Data/synchrony/all/broadband_fill_intermediate"
SYNCHRONY_BROADBAND_FILL_DIRECTORY = "../../Data/synchrony/all/broadband_fill"


# In[ ]:


def find_nan_segments(arr, min_length=240):
    nan_segments = []
    start_index = None
    nan_count = 0

    for i, value in enumerate(arr):
        if np.isnan(value):
            nan_count += 1
            if start_index is None:
                start_index = i
        else:
            if nan_count >= min_length:
                nan_segments.append((start_index, i - 1))
            start_index = None
            nan_count = 0

    # Check for the case where the array ends with a NaN segment
    if nan_count >= min_length:
        nan_segments.append((start_index, len(arr) - 1))

    return nan_segments


# In[ ]:


print("Using Carlos session")
with open("agu_ieeglogin.bin", "r") as f:
    session = Session("aguilac", f.read())


# In[ ]:


# Iterate through all files in SYNCHRONY_BROADBAND_DIRECTORY
for filename in os.listdir(SYNCHRONY_BROADBAND_DIRECTORY):
    # load only .npy files
    if filename.endswith(".npy"):
        # Filenames are in the format of HUP_{patient_id}.npy
        hup_id = filename.split("_")[1].split(".")[0]
        if int(hup_id) % 2 == 0:
            continue
        # Load the data
        og_data = np.load(os.path.join(SYNCHRONY_BROADBAND_DIRECTORY, filename))
        # Find NaN segments
        nan_segments = find_nan_segments(og_data)
        # If the first element of the first tuple in nan_segments is 0, then delete the first tuple
        if nan_segments[0][0] == 0:
            nan_segments = nan_segments[1:]
        if len(nan_segments) == 1:
            print(f"Filling incomplete data for HUP {hup_id}...")
            print(nan_segments)
            for segment in nan_segments:
                print(f"Segment: {segment}")
                segment_start = segment[0]
                segment_end = segment[1]
                if os.path.exists(
                    os.path.join(
                        SYNCHRONY_BROADBAND_FILL_DIRECTORY,
                        f"HUP_{hup_id}_{segment_start}_{segment_end}.npy",
                    )
                ):
                    print(
                        f"HUP_{hup_id}_{segment_start}_{segment_end}.npy exists, skip..."
                    )
                    continue
                dataset_name = f"HUP{hup_id}_phaseII"
                dataset = session.open_dataset(dataset_name)

                all_channel_labels = np.array(dataset.get_channel_labels())
                channel_labels_to_download = all_channel_labels[
                    electrode_selection(all_channel_labels)
                ]

                duration_usec = dataset.get_time_series_details(
                    channel_labels_to_download[0]
                ).duration
                duration_hours = int(duration_usec / 1000000 / 60 / 60)
                enlarged_duration_hours = duration_hours + 24

                print(f"Opening {dataset_name} with duration {duration_hours} hours")

                # Calculate the total number of 2-minute intervals in the enlarged duration
                total_intervals = enlarged_duration_hours * 30  # 60min/hour / 2min = 30

                synchrony_broadband_vector_to_save = np.full(total_intervals, np.nan)

                # Loop through each 2-minute interval
                for interval in range(segment_start, segment_end + 1):
                    print(
                        f"Getting iEEG data for interval {interval} out of {total_intervals}"
                    )
                    duration_usec = 1.2e8  # 2 minutes
                    start_time_usec = (
                        interval * 2 * 60 * 1e6
                    )  # 2 minutes in microseconds
                    stop_time_usec = start_time_usec + duration_usec

                    try:
                        ieeg_data, fs = get_iEEG_data(
                            "aguilac",
                            "agu_ieeglogin.bin",
                            dataset_name,
                            start_time_usec,
                            stop_time_usec,
                            channel_labels_to_download,
                        )
                        fs = int(fs)
                    except Exception as e:
                        # handle the exception
                        print(f"Error: {e}")
                        break

                    # Drop rows that has any nan
                    ieeg_data = ieeg_data.dropna(axis=0, how="any")
                    if ieeg_data.empty:
                        print("Empty dataframe after dropping nan, skip...")
                        continue

                    good_channels_res = detect_bad_channels_optimized(
                        ieeg_data.to_numpy(), fs
                    )
                    good_channel_indicies = good_channels_res[0]
                    good_channel_labels = channel_labels_to_download[
                        good_channel_indicies
                    ]
                    ieeg_data = ieeg_data[good_channel_labels].to_numpy()

                    # Check if ieeg_data is empty after dropping bad channels
                    if ieeg_data.size == 0:
                        print("Empty dataframe after dropping bad channels, skip...")
                        continue

                    ieeg_data = common_average_montage(ieeg_data)

                    # Apply the filters directly on the DataFrame
                    ieeg_data = notch_filter(ieeg_data, 59, 61, fs)

                    ##############################
                    # Calculate synchrony (broadband)
                    ##############################
                    _, R = calculate_synchrony(ieeg_data.T)
                    synchrony_broadband_vector_to_save[interval] = R

                    print(f"Finished calculating synchrony for interval {interval}")

                ##############################
                # Save the synchrony output
                ##############################
                np.save(
                    os.path.join(
                        SYNCHRONY_BROADBAND_FILL_DIRECTORY,
                        f"HUP_{hup_id}_{segment_start}_{segment_end}.npy",
                    ),
                    synchrony_broadband_vector_to_save,
                )
                print(f"Saved HUP_{hup_id}_{segment_start}_{segment_end}.npy")
