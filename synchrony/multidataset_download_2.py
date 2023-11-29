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


SYNCHRONY_BROADBAND_MULTI_DS_DIRECTORY = (
    "../../Data/synchrony/all/broadband_multi_dataset"
)


# In[ ]:


# Load ../../Data/multi_dataset_batches.csv as a pandas dataframe
df = pd.read_csv("../../Data/multi_dataset_batches.csv")
# add a new column called has_multi_sampling_rate
df["has_multi_sampling_rate"] = False
df


# In[ ]:


# Implement a function that takes in a stringified list of integers and returns a list of integers
def string_to_list(string):
    """
    Convert a stringified list of integers to a list of integers

    Parameters
    ----------
    string: str
        A stringified list of integers

    Returns
    -------
    list
        A list of integers
    """
    # YOUR CODE HERE
    return [int(x) for x in string.strip("[]").split(",")]


# In[ ]:


# for each row in the dataframe, load sampling_rate into a list. if there are multiple unique sampling rates, mark has_multi_sampling_rate as True
for index, row in df.iterrows():
    sampling_rate = string_to_list(row["sampling_rate"])
    if len(set(sampling_rate)) > 1:
        df.at[index, "has_multi_sampling_rate"] = True


# In[ ]:


# only keep rows with has_multi_sampling_rate as False
multiple_sample_rate_df = df[df["has_multi_sampling_rate"] == True]
# Reset index
multiple_sample_rate_df = multiple_sample_rate_df.reset_index(drop=True)
# Drop the batch column
multiple_sample_rate_df = multiple_sample_rate_df.drop(columns=["batch"])
multiple_sample_rate_df


# In[ ]:


# only keep rows with has_multi_sampling_rate as False
single_sample_rate_df = df[df["has_multi_sampling_rate"] == False]
# Reset index
single_sample_rate_df = single_sample_rate_df.reset_index(drop=True)
# Drop the batch column
single_sample_rate_df = single_sample_rate_df.drop(columns=["batch"])
single_sample_rate_df


# In[ ]:


def assign_batches(df, column):
    # Sort by the specified column
    sorted_df = df.sort_values(by=column)

    # Calculate total and target size for each batch
    total_size = sorted_df[column].sum()
    target_per_batch = total_size / 4

    # Initialize batch column
    sorted_df["batch"] = 0
    current_batch = 1
    current_sum = 0

    # Iteratively assign batch numbers
    for index, row in sorted_df.iterrows():
        if current_sum + row[column] > target_per_batch and current_batch < 4:
            current_batch += 1
            current_sum = 0
        sorted_df.at[index, "batch"] = current_batch
        current_sum += row[column]

    return sorted_df


# Assign batches to single_sample_rate_df
single_sample_rate_df = assign_batches(single_sample_rate_df, "size_estimate")


# In[ ]:


single_sample_rate_df


# In[ ]:


print("Using Carlos session")
with open("agu_ieeglogin.bin", "r") as f:
    session = Session("aguilac", f.read())


# In[ ]:


batch = single_sample_rate_df[single_sample_rate_df["batch"] == 2].reset_index(
    drop=True
)
batch


# In[ ]:


for index, row in batch.iterrows():
    hup_id = row["hup_id"]
    num_datasets = row["num_datasets"]
    print(f"HUP {hup_id} has {num_datasets} datasets")
    for ds_index in range(1, num_datasets + 1):
        # Check if the file with name f"HUP_{hup_id}_ds_{ds_index}.npy" in SYNCHRONY_BROADBAND_MULTI_DS_DIRECTORY exists
        if os.path.exists(
            os.path.join(
                SYNCHRONY_BROADBAND_MULTI_DS_DIRECTORY,
                f"HUP_{hup_id}_ds_{ds_index}.npy",
            )
        ):
            print(f"HUP_{hup_id}_ds_{ds_index}.npy exists, skip...")
            continue
        dataset_name = f"HUP{hup_id}_phaseII_D0{ds_index}"
        print(f"Opening {dataset_name}")
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
        for interval in range(total_intervals):
            print(f"Getting iEEG data for interval {interval} out of {total_intervals}")
            duration_usec = 1.2e8  # 2 minutes
            start_time_usec = interval * 2 * 60 * 1e6  # 2 minutes in microseconds
            stop_time_usec = start_time_usec + duration_usec

            try:
                ieeg_data, fs = get_iEEG_data(
                    "pattnaik",
                    "pat_ieeglogin.bin",
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
                SYNCHRONY_BROADBAND_MULTI_DS_DIRECTORY,
                f"HUP_{hup_id}_ds_{ds_index}.npy",
            ),
            synchrony_broadband_vector_to_save,
        )
        print(f"Saved HUP_{hup_id}_ds_{ds_index}.npy")
