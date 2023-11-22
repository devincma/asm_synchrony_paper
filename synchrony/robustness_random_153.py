#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from ieeg.auth import Session

from get_iEEG_data import *
from iEEG_helper_functions import *


# In[ ]:


SYNCHRONY_ROBUSTNESS_DIRECTORY = "../../Data/synchrony/all/robustness"
hup_ids_to_compute = [153]


# In[ ]:


# Load HUP_implant_dates.xlsx
nina_patients_df = pd.read_excel("../../Data/HUP_implant_dates.xlsx")
# Make the hup_id column integers
nina_patients_df["hup_id"] = nina_patients_df["hup_id"].astype(int)
nina_patients_df


# In[ ]:


# Only keep the rows in nina_patients_df that have hup_ids not in completed_hup_ids
nina_patients_df = nina_patients_df[nina_patients_df["hup_id"].isin(hup_ids_to_compute)]
# Reset the index
nina_patients_df = nina_patients_df.reset_index(drop=True)
nina_patients_df


# In[ ]:


print("Using Carlos session")
with open("agu_ieeglogin.bin", "r") as f:
    session = Session("aguilac", f.read())


# In[ ]:


electrode_counts = list(range(10, 101, 10))

# Iterate through every row in batch
for index, row in nina_patients_df.iterrows():
    hup_id = row["hup_id"]
    dataset_name = f"HUP{hup_id}_phaseII_D01"

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

    # Store synchrony vectors for different electrode counts in a dictionary for convenience
    synchrony_vectors_to_save = {}

    # Initialize arrays for each electrode count
    for count in electrode_counts:
        synchrony_vectors_to_save[count] = np.full(total_intervals, np.nan)

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

        # Drop rows that has any nan
        ieeg_data = ieeg_data.dropna(axis=0, how="any")
        if ieeg_data.empty:
            print("Empty dataframe after dropping nan, skip...")
            continue

        good_channels_res = detect_bad_channels_optimized(ieeg_data.to_numpy(), fs)
        good_channel_indicies = good_channels_res[0]
        good_channel_labels = channel_labels_to_download[good_channel_indicies]

        # If the len of good_channel_labels is less than 2, skip
        if len(good_channel_labels) < 2:
            print("Less than 2 good channels, skip...")
            continue

        for count in electrode_counts:
            # Randomly select electrodes given the constraints
            num_to_select = min(len(good_channel_labels), count)
            print(
                f"Selecting {num_to_select} electrodes out of {len(good_channel_labels)} good channels"
            )
            selected_labels = np.random.choice(
                good_channel_labels, num_to_select, replace=False
            )

            selected_ieeg_data = ieeg_data[selected_labels].to_numpy()

            # Check if selected_ieeg_data is empty after dropping bad channels
            if selected_ieeg_data.size == 0:
                print(f"Empty dataframe after selecting {count} electrodes, skip...")
                continue

            selected_ieeg_data = common_average_montage(selected_ieeg_data)
            selected_ieeg_data = notch_filter(selected_ieeg_data, 59, 61, fs)

            # Calculate synchrony for the selected electrodes
            _, R = calculate_synchrony(selected_ieeg_data.T)
            synchrony_vectors_to_save[count][interval] = R

    # Save the synchrony output for each electrode count
    for count in electrode_counts:
        np.save(
            os.path.join(
                SYNCHRONY_ROBUSTNESS_DIRECTORY, f"HUP_{hup_id}_D01_random_{count}.npy"
            ),
            synchrony_vectors_to_save[count],
        )
        print(f"Saved synchrony output for HUP {hup_id} with {count} electrodes")
