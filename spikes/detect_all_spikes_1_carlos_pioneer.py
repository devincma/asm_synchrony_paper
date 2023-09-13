#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from ieeg.auth import Session

from get_iEEG_data import *
from spike_detector import *
from spike_morphology import *
from iEEG_helper_functions import *


# In[ ]:


SPIKES_OUTPUT_DIR = "../../Data/spikes/devin_spikes_new/"


# In[ ]:


good_hup_ids_for_spike_detector = np.load("good_hup_ids_for_spike_detector.npy")
good_hup_ids_for_spike_detector


# In[ ]:


good_hup_ids_for_spike_detector.shape


# In[ ]:


# Load HUP_implant_dates.xlsx
nina_patients_df = pd.read_excel("../../Data/HUP_implant_dates.xlsx")
# Make the hup_id column integers
nina_patients_df["hup_id"] = nina_patients_df["hup_id"].astype(int)
nina_patients_df


# In[ ]:


# Add a boolean column in nina_patients_df called is_single_dataset and make it True if IEEG_Portal_Number ends with "phaseII"
nina_patients_df["is_single_dataset"] = nina_patients_df[
    "IEEG_Portal_Number"
].str.endswith("phaseII")
# Add a boolean column in nina_patients_df called is_good_for_spike_detector and make it True if the row's hup_id is in good_hup_ids_for_spike_detector
nina_patients_df["is_good_for_spike_detector"] = nina_patients_df["hup_id"].isin(
    good_hup_ids_for_spike_detector
)
nina_patients_df


# In[ ]:


# Drop the rows in nina_patients_df where is_single_dataset is False
nina_patients_df = nina_patients_df[nina_patients_df.is_single_dataset == True]
# Drop the rows in nina_patients_df where is_good_for_spike_detector is False
nina_patients_df = nina_patients_df[nina_patients_df.is_good_for_spike_detector == True]
# Sort by hup_id in ascending order
nina_patients_df = nina_patients_df.sort_values(by=["hup_id"], ascending=True)
# Drop columns Implant_Date, implant_time, Explant_Date, weight_kg
nina_patients_df = nina_patients_df.drop(
    columns=["Implant_Date", "implant_time", "Explant_Date", "weight_kg"]
)
# Reset index
nina_patients_df = nina_patients_df.reset_index(drop=True)
nina_patients_df


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 6 == 0].reset_index(drop=True)


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 6 == 1].reset_index(drop=True)


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 6 == 2].reset_index(drop=True)


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 6 == 3].reset_index(drop=True)


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 6 == 4].reset_index(drop=True)


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 6 == 5].reset_index(drop=True)


# ## Select a batch

# In[ ]:


batch = nina_patients_df[nina_patients_df["hup_id"] % 6 == 1].reset_index(drop=True)
batch


# In[ ]:


# print("Using Devin session")
# with open("dma_ieeglogin.bin", "r") as f:
#     session = Session("dma", f.read())
print("Using Carlos session")
with open("agu_ieeglogin.bin", "r") as f:
    session = Session("aguilac", f.read())


# In[ ]:


# Iterate through every row in batch
for index, row in batch.iterrows():
    hup_id = row["hup_id"]
    dataset_name = row["IEEG_Portal_Number"]

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

    # Loop through each 2-minute interval
    for interval in range(total_intervals):
        print(f"Getting iEEG data for interval {interval} out of {total_intervals}")
        duration_usec = 1.2e8  # 2 minutes
        start_time_usec = interval * 2 * 60 * 1e6  # 2 minutes in microseconds
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

        # Check if ieeg_data dataframe is all NaNs
        if ieeg_data.isnull().values.all():
            print("Empty dataframe after download, skip...")
            continue

        good_channels_res = detect_bad_channels_optimized(ieeg_data.to_numpy(), fs)
        good_channel_indicies = good_channels_res[0]
        good_channel_labels = channel_labels_to_download[good_channel_indicies]
        ieeg_data = ieeg_data[good_channel_labels].to_numpy()

        # Check if ieeg_data is empty after dropping bad channels
        if ieeg_data.size == 0:
            print("Empty dataframe after artifact rejection, skip...")
            continue

        ieeg_data = common_average_montage(ieeg_data)

        # Apply the filters directly on the DataFrame
        ieeg_data = notch_filter(ieeg_data, 59, 61, fs)
        ieeg_data = bandpass_filter(ieeg_data, 1, 70, fs)

        ##############################
        # Detect spikes
        ##############################

        spike_output = spike_detector(
            data=ieeg_data,
            fs=fs,
            electrode_labels=good_channel_labels,
        )
        if len(spike_output) == 0:
            print("No spikes detected, skip saving...")
            continue
        else:
            print(f"Detected {len(spike_output)} spikes")

        ##############################
        # Extract spike morphologies
        ##############################
        # Preallocate the result array
        spike_output_to_save = np.empty((spike_output.shape[0], 16), dtype=object)
        spike_output_to_save[:, :] = np.NaN  # Fill with NaNs

        for i, spike in enumerate(spike_output):
            peak_index = int(spike[0])
            channel_index = int(spike[1])
            channel_label = spike[2]

            # Fill the first two columns with peak_index and channel_index
            spike_output_to_save[i, 0] = peak_index
            spike_output_to_save[i, 1] = channel_index
            spike_output_to_save[i, 2] = channel_label

            spike_signal = ieeg_data[
                peak_index - 1000 : peak_index + 1000, channel_index
            ]

            try:
                (
                    basic_features,
                    advanced_features,
                    is_valid,
                    bad_reason,
                ) = extract_spike_morphology(spike_signal)

                if is_valid:
                    # Fill the rest of the columns with computed features
                    spike_output_to_save[i, 3:8] = basic_features
                    spike_output_to_save[i, 8:16] = advanced_features
            except Exception as e:
                print(f"Error extracting spike features: {e}")
                continue

        ##############################
        # Save the spike output
        ##############################
        np.save(
            os.path.join(SPIKES_OUTPUT_DIR, f"{dataset_name}_{interval}.npy"),
            spike_output_to_save,
        )
        print(f"Saved spike output for interval {interval} for HUP {hup_id}")
