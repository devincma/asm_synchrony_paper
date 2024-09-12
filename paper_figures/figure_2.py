#!/usr/bin/env python
# coding: utf-8

# # Figure 2 (Baseline Analysis)

# In[2]:


import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


# In[3]:


# Replace with the path to where you store data
DATA_PATH = "../../cnt-data/"


# In[4]:


# This is the example patient's HUP ID
patient_hup_id = 142

# Load in the patient's "giant table"
hourly_patient_features_df = pd.read_csv(
    os.path.join(DATA_PATH, f"giant_new_tables/HUP_{str(patient_hup_id)}.csv")
)


# ## Figure 2 (C)

# In[5]:


# Assuming you have the hourly_patient_features_df from somewhere above
seizure_indices = hourly_patient_features_df[
    hourly_patient_features_df["had_seizure"] >= 1
].index

valid_seizure_indices = [
    idx
    for i, idx in enumerate(seizure_indices)
    if i == 0 or (seizure_indices[i] - seizure_indices[i - 1]) >= 120
]

avg_synchronies = []
avg_med_loads = []

for idx in valid_seizure_indices:
    seizure_time = hourly_patient_features_df.loc[idx, "emu_minute"]

    window = hourly_patient_features_df[
        (hourly_patient_features_df["emu_minute"] >= seizure_time - 120)
        & (hourly_patient_features_df["emu_minute"] < seizure_time - 20)
    ]

    avg_synchrony = np.nanmean(window["synchrony_broadband"])
    avg_med_load = np.nanmean(window["med_sum_no_lorazepam_raw"])

    avg_synchronies.append(avg_synchrony)
    avg_med_loads.append(avg_med_load)

if avg_med_loads and avg_synchronies:
    plt.figure(figsize=(3, 3))  # Add this line to create a standalone figure
    sns.regplot(
        x=avg_med_loads,
        y=avg_synchronies,
        scatter=True,
        line_kws={"color": "red"},
        ci=None,
        color="red",
    )

    # Label each dot with its respective seizure number
    for i, (x, y) in enumerate(zip(avg_med_loads, avg_synchronies)):
        plt.text(
            x, y, f"Seizure {i+1}", fontsize=10, ha="center", va="bottom", color="red"
        )

    plt.xlabel("Average ASM Load")
    plt.ylabel("Average Baseline Synchrony")
    # plt.title(f"Baseline Synchrony vs ASM")
    plt.show()  # Add this line to display the plot


# ## Figure 2 (D)

# In[6]:


def get_patient_hup_ids(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter out files based on the given pattern and extract patient_hup_id as integers
    patient_hup_ids = [
        int(f.split("_")[1].split(".")[0]) for f in files if f.startswith("HUP_")
    ]

    return patient_hup_ids


TABLES_PATH = os.path.join(DATA_PATH, "giant_new_tables")
completed_hup_ids = get_patient_hup_ids(TABLES_PATH)
completed_hup_ids.sort()


# In[8]:


completed_hup_ids


# In[12]:


def get_seizure_data(hourly_patient_features_df):
    """
    For a given hourly_patient_features_df, return a list of tuples.
    Each tuple contains avg_med_load and avg_synchrony for each seizure.
    """
    seizure_indices = hourly_patient_features_df[
        hourly_patient_features_df["had_seizure"] >= 1
    ].index

    valid_seizure_indices = [
        idx
        for i, idx in enumerate(seizure_indices)
        if i == 0 or (seizure_indices[i] - seizure_indices[i - 1]) >= 5 * 60
    ]

    seizure_data = []

    for idx in valid_seizure_indices:
        seizure_time = hourly_patient_features_df.loc[idx, "emu_minute"]

        window = hourly_patient_features_df[
            (hourly_patient_features_df["emu_minute"] >= seizure_time - 60)
            & (hourly_patient_features_df["emu_minute"] < seizure_time)
        ]
        if len(window["synchrony_broadband"]) == 0 or np.all(
            np.isnan(window["synchrony_broadband"])
        ):
            continue  # Skip the rest of the current iteration

        avg_synchrony = np.nanmedian(window["synchrony_broadband"])
        avg_med_load = np.nanmedian(window["med_sum_no_lorazepam_raw"])

        if avg_synchrony < 0.6:
            seizure_data.append((avg_med_load, avg_synchrony))

    return seizure_data


# Accumulate data from all patients
all_seizures_data = []

for patient_hup_id in completed_hup_ids:
    hourly_patient_features_df = pd.read_csv(
        os.path.join(TABLES_PATH, f"HUP_{str(patient_hup_id)}.csv")
    )
    all_seizures_data.extend(get_seizure_data(hourly_patient_features_df))

# Scatter plot
plt.figure(figsize=(6, 6))
avg_med_loads, avg_synchronies = zip(*all_seizures_data)

# Convert tuples to lists
avg_med_loads_list = list(avg_med_loads)
avg_synchronies_list = list(avg_synchronies)

# Fit an OLS regression model
X = sm.add_constant(avg_med_loads_list)  # Adding a constant for the intercept
model = sm.OLS(avg_synchronies_list, X).fit()

# Print out the statistics
print(model.summary())

sns.regplot(
    x=avg_med_loads_list,
    y=avg_synchronies_list,
    scatter=True,
    line_kws={"color": "red"},
    ci=None,
)

plt.xlabel("Average ASM Load")
plt.ylabel("Average Baseline Synchrony")
plt.grid(True)
plt.show()
