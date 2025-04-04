#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from ieeg.auth import Session

from get_iEEG_data import *
from iEEG_helper_functions import *


# In[2]:


SPIKES_OUTPUT_DIR = "../../Data/spikes/devin_spikes/"
SYNCHRONY_broadband_DIRECTORY = "../../Data/synchrony/all/broadband"


# In[3]:


# Load HUP_implant_dates.xlsx
nina_patients_df = pd.read_excel("../../Data/HUP_implant_dates.xlsx")
# Make the hup_id column integers
nina_patients_df["hup_id"] = nina_patients_df["hup_id"].astype(int)
nina_patients_df


# ## Already completed

# In[4]:


incomplete_hup_ids = [
    143,
    146,
    155,
    157,
    158,
    161,
    163,
    164,
    165,
    171,
    182,
    188,
    206,
    210,
]
incomplete_hup_ids.sort()
print(incomplete_hup_ids)
len(incomplete_hup_ids)


# In[5]:


# Only keep the rows in nina_patients_df that have hup_ids not in completed_hup_ids
nina_patients_df = nina_patients_df[nina_patients_df["hup_id"].isin(incomplete_hup_ids)]
# Reset the index
nina_patients_df = nina_patients_df.reset_index(drop=True)
nina_patients_df


# In[6]:


# Add a boolean column in nina_patients_df called is_single_dataset and make it True if IEEG_Portal_Number ends with "phaseII"
nina_patients_df["is_single_dataset"] = nina_patients_df[
    "IEEG_Portal_Number"
].str.endswith("phaseII")
nina_patients_df


# In[7]:


# Drop the rows in nina_patients_df where is_single_dataset is False
nina_patients_df = nina_patients_df[nina_patients_df.is_single_dataset == True]
# Sort by hup_id in ascending order
nina_patients_df = nina_patients_df.sort_values(by=["hup_id"], ascending=True)
# Drop columns Implant_Date, implant_time, Explant_Date, weight_kg
nina_patients_df = nina_patients_df.drop(
    columns=["Implant_Date", "implant_time", "Explant_Date", "weight_kg"]
)
# Reset index
nina_patients_df = nina_patients_df.reset_index(drop=True)
nina_patients_df


# In[8]:


nina_patients_df[nina_patients_df["hup_id"] % 2 == 0].reset_index(drop=True)


# In[9]:


nina_patients_df[nina_patients_df["hup_id"] % 2 == 1].reset_index(drop=True)


# ## Select a batch

# In[10]:


batch = nina_patients_df[nina_patients_df["hup_id"] % 2 == 1].reset_index(drop=True)
batch


# In[ ]:


print("Using Carlos session")
with open("agu_ieeglogin.bin", "r") as f:
    session = Session("aguilac", f.read())


# In[ ]:


# Iterate through every row in batch
for index, row in batch.iterrows():
    hup_id = row["hup_id"]
    dataset_name = row["IEEG_Portal_Number"]
    print(dataset_name)

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
        os.path.join(SYNCHRONY_broadband_DIRECTORY, f"HUP_{hup_id}.npy"),
        synchrony_broadband_vector_to_save,
    )
    print(f"Saved synchrony output for HUP {hup_id}")

