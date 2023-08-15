#!/usr/bin/env python
# coding: utf-8

# In[3]:


# !rm -rf __pycache__
import numpy as np
import os, pickle
import pandas as pd
from scipy.signal import hilbert
from iEEG_helper_functions import *

IEEG_DIRECTORY = "../../Data/ieeg/all/2_min"
SYNCHRONY_60_100_DIRECTORY = "../../Data/synchrony/hourly/60_100"


# In[2]:


def calculate_synchrony(time_series):
    """
    Calculate the Kuramoto order parameter for a set of time series
    Args:
        time_series (np.array): 2D array where each row is a time series
    Returns:
        np.array: Kuramoto order parameter for each time point
    """
    # Extract the number of time series and the number of time points
    N, _ = time_series.shape
    # Apply the Hilbert Transform to get an analytical signal
    analytical_signals = hilbert(time_series)
    assert analytical_signals.shape == time_series.shape
    # Extract the instantaneous phase for each time series using np.angle
    phases = np.angle(analytical_signals, deg=False)
    assert phases.shape == time_series.shape
    # Compute the Kuramoto order parameter for each time point
    # 1j*1j == -1
    r_t = np.abs(np.sum(np.exp(1j * phases), axis=0)) / N
    R = np.mean(r_t)
    return r_t, R


# def calculate_entropy(synchrony, num_bins=24):
#     # Calculate the probability distribution by binning the synchrony values
#     hist, _ = np.histogram(synchrony, bins=num_bins)
#     probabilities = hist / np.sum(hist)

#     # Calculate the entropy
#     entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
#     return entropy


# In[3]:


nina_patient_hup_ids = pd.read_excel("../../Data/HUP_implant_dates.xlsx")
nina_patient_hup_ids = nina_patient_hup_ids["hup_id"].to_numpy()
nina_patient_hup_ids


# In[4]:


# Create a mapping between patient ids and the index of the patient in the patients_df dataframe
patient_hup_id_to_index = {}
for i, patient_id in enumerate(nina_patient_hup_ids):
    patient_hup_id_to_index[patient_id] = i
# patient_hup_id_to_index


# In[5]:


ieeg_offset_row1_df = pd.read_excel("../../Data/ieeg_offset/row_1.xlsx", header=None)
ieeg_offset_row2_df = pd.read_excel("../../Data/ieeg_offset/row_2.xlsx", header=None)
ieeg_offset_row3_df = pd.read_excel("../../Data/ieeg_offset/row_3.xlsx", header=None)


# In[6]:


# Load master_elecs.csv from ./data/
master_elecs_df = pd.read_csv("../../Data/master_elecs.csv")

# only take the numbers in rid column
master_elecs_df["rid"] = master_elecs_df["rid"].str.extract("(\d+)", expand=False)
master_elecs_df["rid"] = master_elecs_df["rid"].astype(int)

# Drop mni_x, mni_y, mni_z, mm_x, mm_y, mm_z columns
master_elecs_df = master_elecs_df.drop(
    columns=["mni_x", "mni_y", "mni_z", "mm_x", "mm_y", "mm_z"]
)

master_elecs_df


# In[7]:


# Load rid_hup_table.csv from ./data/
rid_hup_table_df = pd.read_csv("../../Data/rid_hup_table.csv")
# Drop the t3_subject_id and ieegportalsubjno columns
rid_hup_table_df = rid_hup_table_df.drop(columns=["t3_subject_id", "ieegportalsubjno"])
rid_hup_table_df


# In[8]:


# Create an empty dictionary to store all the data
data_dict = {"dataset_name": [], "max_hour": [], "sample_rate": [], "hup_id": []}

# Iterate through the directory
for filename in os.listdir(IEEG_DIRECTORY):
    if filename.endswith(".pkl"):  # Only process .pkl files
        # Split the filename to get the dataset_name, hour, and sample_rate
        parts = filename.split("_")
        dataset_name = "_".join(parts[:-4])  # Exclude the '_hr' from the dataset_name
        hour = int(parts[-3])
        fs = int(parts[-1].split(".")[0])

        # Extract hup_id from dataset_name
        hup_id = dataset_name.split("_")[0].split("HUP")[1]

        # If the dataset_name is already in the dictionary, update the max_hour
        if dataset_name in data_dict["dataset_name"]:
            index = data_dict["dataset_name"].index(dataset_name)
            data_dict["max_hour"][index] = max(data_dict["max_hour"][index], hour)
        else:
            # Else, add the dataset_name, hour, sample_rate and hup_id to the dictionary
            data_dict["dataset_name"].append(dataset_name)
            data_dict["max_hour"].append(hour)
            data_dict["sample_rate"].append(fs)
            data_dict["hup_id"].append(hup_id)

# Create a DataFrame from the dictionary
datasets_df = pd.DataFrame(data_dict)
# Make max_hour and sample_rate and hup_id integers
datasets_df["max_hour"] = datasets_df["max_hour"].astype(int)
datasets_df["sample_rate"] = datasets_df["sample_rate"].astype(int)
datasets_df["hup_id"] = datasets_df["hup_id"].astype(int)
# Sort by hup_id
datasets_df = datasets_df.sort_values(by=["hup_id"])
# Reset the index
datasets_df = datasets_df.reset_index(drop=True)
# Create a column called max_hour_count that is the max_hour + 1
datasets_df["max_hour_count"] = datasets_df["max_hour"] + 1
datasets_df


# In[9]:


ids = datasets_df["hup_id"].unique()
odd_ids = ids[ids % 2 != 0]


# In[ ]:


for patient_hup_id in odd_ids:
    # Find the value of record_id in rid_hup_table_df where hupsubjno == patient_hup_id
    patient_rid = rid_hup_table_df[rid_hup_table_df["hupsubjno"] == patient_hup_id][
        "record_id"
    ].values[0]
    # Get the row in datasets_df corresponding to the patient_hup_id
    rows_df = datasets_df[datasets_df["hup_id"] == patient_hup_id]
    # Sort rows_df by dataset_name
    rows_df = rows_df.sort_values(by=["dataset_name"])
    rows_df = rows_df.reset_index(drop=True)
    patient_electrodes_df = master_elecs_df.loc[master_elecs_df["rid"] == patient_rid]
    print(f"HUP {patient_hup_id}, rid {patient_rid}")

    # Add up all the max_hours for rows_df
    total_max_hour_count = rows_df["max_hour_count"].sum()

    ##########################################
    # Create empty vectors to save the data
    ##########################################
    synchrony_60_100_vector_to_save = np.zeros(total_max_hour_count)
    current_hour = 0

    for dataset_idx, dataset_row in rows_df.iterrows():
        # Get the dataset_name, max_hour, and sample_rate
        dataset_name = dataset_row["dataset_name"]
        max_hour_count = dataset_row["max_hour_count"]
        fs = dataset_row["sample_rate"]
        print(dataset_name)

        for hour in range(max_hour_count):
            # Get the filename
            filename = f"{dataset_name}_hr_{hour}_fs_{fs}.pkl"
            # Get the full path to the file
            full_path = os.path.join(IEEG_DIRECTORY, filename)

            # Load the data
            try:
                with open(full_path, "rb") as f:
                    ieeg_data = pickle.load(f)
            except FileNotFoundError:
                print(f"Skipping {hour} for {dataset_name}")
                synchrony_60_100_vector_to_save[current_hour] = np.nan
                current_hour += 1
                continue

            print(
                f"Processing hour {hour} in {dataset_name}, that's hour {current_hour} out of {total_max_hour_count} for HUP {patient_hup_id}"
            )

            try:
                all_channel_labels = ieeg_data.columns.values.astype(str)
                label_idxs = electrode_selection(all_channel_labels)
                labels = all_channel_labels[label_idxs]
                ieeg_data = ieeg_data[labels]
                good_channels_res = detect_bad_channels_optimized(
                    ieeg_data.to_numpy(), fs
                )
                good_channel_indicies = good_channels_res[0]
                good_labels = labels[good_channel_indicies]
                ieeg_data = ieeg_data[good_labels]

                ieeg_data = common_average_montage(ieeg_data)

                # Broadband
                ieeg_data = pd.DataFrame(notch_filter(ieeg_data.values, 59, 61, fs))

                # 60-100 Hz
                ieeg_data = pd.DataFrame(bandpass_filter(ieeg_data.values, 60, 100, fs))
                _, R = calculate_synchrony((ieeg_data.T).to_numpy())
                synchrony_60_100_vector_to_save[current_hour] = R

                # Increment current_hour
                current_hour += 1

            except:
                print(f"Skipping {hour} for {dataset_name} due to unknown error")
                synchrony_60_100_vector_to_save[current_hour] = np.nan
                current_hour += 1
                continue

    ##########################################
    # Save files
    ##########################################
    np.save(
        f"{SYNCHRONY_60_100_DIRECTORY}/HUP_{patient_hup_id}.npy",
        synchrony_60_100_vector_to_save,
    )


# In[ ]:




