#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys, random
import numpy as np
import pandas as pd
from ieeg.auth import Session

from get_iEEG_data import *
from spike_detector import *
from iEEG_helper_functions import *


# In[ ]:


SPIKES_OUTPUT_DIR = "../../../Data/spikes/devin_spikes_october/"
RANDOMLY_SAVE_CLIPS_DIR = "../../../Data/spikes/randomly_chosen_ieeg_clips/"
LOGS_DIR = "../../../Data/spikes/logs/"


# ## Patient selection

# In[ ]:


good_hup_ids_for_spike_detector = np.load("../good_hup_ids_for_spike_detector.npy")
good_hup_ids_for_spike_detector


# In[ ]:


# Load HUP_implant_dates.xlsx
nina_patients_df = pd.read_excel("../../../Data/HUP_implant_dates.xlsx")
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


# ## Helper functions

# In[ ]:


def format_channels(channel_array):
    formatted_array = []
    for label in channel_array:
        if label == "PZ":
            formatted_array.append(label)
            continue

        # Splitting string into two parts: prefix (letters) and number
        prefix, number = (
            label[: -len([ch for ch in label if ch.isdigit()])],
            label[-len([ch for ch in label if ch.isdigit()]) :],
        )

        # Formatting the number to have two digits
        formatted_number = f"{int(number):02}"

        # Appending prefix and formatted number
        formatted_label = prefix + formatted_number
        formatted_array.append(formatted_label)

    return np.array(formatted_array)


# # Batches

# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 8 == 0].reset_index(drop=True)


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 8 == 1].reset_index(drop=True)


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 8 == 2].reset_index(drop=True)


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 8 == 3].reset_index(drop=True)


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 8 == 4].reset_index(drop=True)


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 8 == 5].reset_index(drop=True)


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 8 == 6].reset_index(drop=True)


# In[ ]:


nina_patients_df[nina_patients_df["hup_id"] % 8 == 7].reset_index(drop=True)


# ## Main loop

# In[ ]:


# Create the structured dtype
dt = np.dtype(
    [
        (
            "channel_label",
            "U10",
        ),
        ("spike_time", "int32"),
        ("spike_sequence", "int32"),
    ]
)


# In[ ]:


print("Using Carlos session")
with open("agu_ieeglogin.bin", "r") as f:
    session = Session("aguilac", f.read())


# In[ ]:


batch_number = 5


# In[ ]:


batch = nina_patients_df[nina_patients_df["hup_id"] % 8 == batch_number].reset_index(
    drop=True
)
batch


# In[20]:


# Construct the full path to the log file
log_file_path = os.path.join(LOGS_DIR, f"log_batch_{str(batch_number)}.txt")

# Ensure the LOGS_DIR exists, if not, create it
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Open the log file in append mode
with open(log_file_path, "a") as file:
    # Redirect the standard output to the log file
    original_stdout = sys.stdout
    sys.stdout = file

    for index, row in batch.iterrows():
        hup_id = row["hup_id"]
        dataset_name = row["IEEG_Portal_Number"]

        print("\n")
        print(f"------Processing HUP {hup_id} with dataset {dataset_name}------")

        ########################################
        # Get the data from IEEG
        ########################################

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

        # Calculate the total number of 1-minute intervals in the enlarged duration
        total_intervals = enlarged_duration_hours * 60  # 60min/hour / 1min = 60

        # Choose 5 unique random intervals before the loop
        chosen_intervals = random.sample(range(total_intervals), 5)
        print(f"Chosen intervals: {chosen_intervals}")

        # Loop through each 2-minute interval
        for interval in range(total_intervals):
            print(
                f"Getting iEEG data for interval {interval} out of {total_intervals} for HUP {hup_id}"
            )
            duration_usec = 6e7  # 1 minute
            start_time_usec = interval * 6e7  # 1 minutes in microseconds
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
                if interval in chosen_intervals:
                    save_path = os.path.join(
                        RANDOMLY_SAVE_CLIPS_DIR,
                        f"ieeg_data_{dataset_name}_{interval}.pkl",
                    )
                    with open(save_path, "wb") as file:
                        pickle.dump(ieeg_data, file)
                    print(f"Saved ieeg_data segment to {save_path}")
            except:
                continue

            # Check if ieeg_data dataframe is all NaNs
            if ieeg_data.isnull().values.all():
                print(f"Empty dataframe after download, skip...")
                continue

            good_channels_res = detect_bad_channels_optimized(ieeg_data.to_numpy(), fs)
            good_channel_indicies = good_channels_res[0]
            good_channel_labels = channel_labels_to_download[good_channel_indicies]
            ieeg_data = ieeg_data[good_channel_labels].to_numpy()

            # Check if ieeg_data is empty after dropping bad channels
            if ieeg_data.size == 0:
                print(f"Empty dataframe after artifact rejection, skip...")
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
            spike_output = spike_output.astype(int)
            actual_number_of_spikes = len(spike_output)

            if actual_number_of_spikes == 0:
                print(f"No spikes detected, skip saving...")
                continue
            else:
                # Map the channel indices to the corresponding good_channel_labels
                channel_labels_mapped = good_channel_labels[spike_output[:, 1]]

                # Create the structured array
                spike_output_to_save = np.array(
                    list(
                        zip(
                            channel_labels_mapped,
                            spike_output[:, 0],
                            spike_output[:, 2],
                        )
                    ),
                    dtype=dt,
                )
                np.save(
                    os.path.join(SPIKES_OUTPUT_DIR, f"{dataset_name}_{interval}.npy"),
                    spike_output_to_save,
                )
                print(
                    f"Saved {actual_number_of_spikes} spikes to {dataset_name}_{interval}.npy"
                )
    # Restore the standard output to its original value
    sys.stdout = original_stdout
