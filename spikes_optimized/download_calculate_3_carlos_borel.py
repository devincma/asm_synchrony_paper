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


SPIKES_OUTPUT_DIR = "../../Data/spikes/devin_spikes/"
SYNCHRONY_60_100_DIRECTORY = "../../Data/synchrony/all/60_100"
SYNCHRONY_100_125_DIRECTORY = "../../Data/synchrony/all/100_125"
SYNCHRONY_broadband_DIRECTORY = "../../Data/synchrony/all/broadband"


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


# Remove the row where hup_id == 143 and hup_id == 210
nina_patients_df = nina_patients_df[nina_patients_df["hup_id"] != 143]
nina_patients_df = nina_patients_df[nina_patients_df["hup_id"] != 210]
# Reset the index
nina_patients_df = nina_patients_df.reset_index(drop=True)
nina_patients_df


# In[ ]:


# Load rid_hup_table.csv from ./data/
rid_hup_table_df = pd.read_csv("../../Data/rid_hup_table.csv")
# Drop the t3_subject_id and ieegportalsubjno columns
rid_hup_table_df = rid_hup_table_df.drop(columns=["t3_subject_id", "ieegportalsubjno"])
# Rename hupsubjno to hup_id
rid_hup_table_df = rid_hup_table_df.rename(columns={"hupsubjno": "hup_id"})
# Sort by hup_id
rid_hup_table_df = rid_hup_table_df.sort_values(by=["hup_id"])
# Only keep rows where hup_id is in nina_patients_df's hup_id column
rid_hup_table_df = rid_hup_table_df[
    rid_hup_table_df["hup_id"].isin(nina_patients_df["hup_id"])
]
# Reset the index
rid_hup_table_df = rid_hup_table_df.reset_index(drop=True)
rid_hup_table_df.head()


# In[ ]:


# Load master_elecs.csv from ./data/
master_elecs_df = pd.read_csv("../../Data/master_elecs.csv")

# only take the numbers in rid column
master_elecs_df["rid"] = master_elecs_df["rid"].str.extract("(\d+)", expand=False)
master_elecs_df["rid"] = master_elecs_df["rid"].astype(int)

# Drop mni_x, mni_y, mni_z, mm_x, mm_y, mm_z columns
master_elecs_df = master_elecs_df.drop(
    columns=["mni_x", "mni_y", "mni_z", "mm_x", "mm_y", "mm_z"]
)
# Rename rid to record_id
master_elecs_df = master_elecs_df.rename(columns={"rid": "record_id"})
# Add a column called hup_id using the table rid_hup_table_df
master_elecs_df = master_elecs_df.merge(rid_hup_table_df, on="record_id", how="left")
# Drop the rows where hup_id is NaN
master_elecs_df = master_elecs_df.dropna(subset=["hup_id"])
# Make hup_id an integer
master_elecs_df["hup_id"] = master_elecs_df["hup_id"].astype(int)
# Sort by hup_id
master_elecs_df = master_elecs_df.sort_values(by=["hup_id"])
# Reset index
master_elecs_df = master_elecs_df.reset_index(drop=True)
master_elecs_df.head()


# In[ ]:


# Only keep rows in nina_patients_df whose hup_id is in master_elecs_df's hup_id column
nina_patients_df = nina_patients_df[
    nina_patients_df["hup_id"].isin(master_elecs_df["hup_id"])
]
# Reset index
nina_patients_df = nina_patients_df.reset_index(drop=True)
nina_patients_df


# ## Select a batch

# In[ ]:


batch = nina_patients_df[nina_patients_df["hup_id"] % 4 == 3].reset_index(drop=True)
batch


# In[ ]:


# def create_pwd_file(username, password, fname=None):
#     if fname is None:
#         fname = "{}_ieeglogin.bin".format(username[:3])
#     with open(fname, "wb") as f:
#         f.write(password.encode())
#     print("-- -- IEEG password file saved -- --")


# create_pwd_file("dma", "mycqEv-pevfo4-roqfan")
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

    synchrony_broadband_vector_to_save = np.full(total_intervals, np.nan)
    synchrony_60_100_vector_to_save = np.full(total_intervals, np.nan)
    synchrony_100_125_vector_to_save = np.full(total_intervals, np.nan)

    # Loop through each 2-minute interval
    for interval in range(total_intervals):
        print(f"Getting iEEG data for interval {interval} out of {total_intervals}")
        duration_usec = 1.2e8  # 2 minutes
        start_time_usec = interval * 2 * 60 * 1e6  # 2 minutes in microseconds
        stop_time_usec = start_time_usec + duration_usec

        try:
            ieeg_data, fs = get_iEEG_data(
                "dma",
                "dma_ieeglogin.bin",
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

        ##############################
        # Calculate synchrony (60-100Hz)
        ##############################
        ieeg_data_60_100 = bandpass_filter(ieeg_data, 60, 100, fs)
        _, R = calculate_synchrony(ieeg_data_60_100.T)
        synchrony_60_100_vector_to_save[interval] = R

        ##############################
        # Calculate synchrony (100-125Hz)
        ##############################
        try:
            ieeg_data_100_125 = bandpass_filter(ieeg_data, 100, 125, fs)
            _, R = calculate_synchrony(ieeg_data_100_125.T)
            synchrony_100_125_vector_to_save[interval] = R
        except Exception as e:
            print(f"Error: {e}")

        print(f"Finished calculating synchrony for interval {interval}")

        ##############################
        # Detect spikes
        ##############################
        ieeg_data_for_spikes = bandpass_filter(ieeg_data, 1, 70, fs)

        spike_output = spike_detector(
            data=ieeg_data_for_spikes,
            fs=fs,
            labels=good_channel_labels,
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
        spike_output_to_save = np.empty((spike_output.shape[0], 15), dtype=np.float64)
        spike_output_to_save[:, :] = np.NaN  # Fill with NaNs

        for i, spike in enumerate(spike_output):
            peak_index, channel_index = spike
            spike_signal = ieeg_data_for_spikes[
                peak_index - 1000 : peak_index + 1000, channel_index
            ]

            # Fill the first two columns with peak_index and channel_index
            spike_output_to_save[i, 0] = peak_index
            spike_output_to_save[i, 1] = channel_index

            try:
                (
                    basic_features,
                    advanced_features,
                    is_valid,
                    bad_reason,
                ) = extract_spike_morphology(spike_signal)

                if is_valid:
                    # Fill the rest of the columns with computed features
                    spike_output_to_save[i, 2:7] = basic_features
                    spike_output_to_save[i, 7:15] = advanced_features
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

    ##############################
    # Save the synchrony output
    ##############################
    np.save(
        os.path.join(SYNCHRONY_broadband_DIRECTORY, f"HUP_{hup_id}.npy"),
        synchrony_broadband_vector_to_save,
    )
    np.save(
        os.path.join(SYNCHRONY_60_100_DIRECTORY, f"HUP_{hup_id}.npy"),
        synchrony_60_100_vector_to_save,
    )
    np.save(
        os.path.join(SYNCHRONY_100_125_DIRECTORY, f"HUP_{hup_id}.npy"),
        synchrony_100_125_vector_to_save,
    )
    print(f"Saved synchrony output for HUP {hup_id}")

