import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, stats
import pandas as pd

def plot_aed_curves(all_dose_curves_plot, all_med_names_plot, all_tHr_plot, aed_ref_ranges_df, patient_id, time_axis, seizure_times, aligned_emu_start_time_hrs):
    #########################################
    # Plot AED dose
    #########################################

    sum_array = []

    # Create two subplots with shared x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(18, 12))

    # Plot dose curves
    for i in range(len(all_dose_curves_plot)):

        # # Lorazepam is a rescue medicine, skip it in dose calculations
        # if all_med_names_plot[i] == "lorazepam":
        #     continue

        med_name = all_med_names_plot[i]
        print(med_name)

        dose_times = all_tHr_plot[i].flatten()

        # Find Avg for medication med_name in aed_ref_ranges_df
        if med_name != "lorazepam":
            ref_range = float(
                aed_ref_ranges_df.loc[aed_ref_ranges_df["Drug"] == med_name, "Avg"].values[
                    0
                ]
            )
        else:
            ref_range = 1
        print(f"{med_name} has ref range {ref_range} mg/L")

        # Normalize dose curve according to the middle of the reference range
        dose = all_dose_curves_plot[i].flatten()
        dose = dose / ref_range

        interp_func = interpolate.interp1d(
            dose_times, dose, bounds_error=False, fill_value=0
        )
        dose_interp = interp_func(time_axis)

        if med_name != "lorazepam":
            print(f"adding {med_name} to sum_array")
            sum_array.append(dose_interp)
            print(f"plotting {med_name}")
            
        ax1.plot(time_axis, dose_interp, label=med_name)

    cumulative_dose_curve = np.sum(sum_array, axis=0)
    # cumulative_dose_curve = cumulative_dose_curve / cumulative_dose_curve.max()
    assert len(cumulative_dose_curve) == len(
        time_axis
    ), "cumulative_dose_curve and time_axis should have the same length"

    # Convert numpy arrays to pandas DataFrame
    df = pd.DataFrame(data={"Time": time_axis, "Value": cumulative_dose_curve})
    # Create a new column for the integer hour
    df["Hour"] = df["Time"].astype(int)

    # Group by hour and compute mean
    hourly_df = df.groupby("Hour").mean()
    # Create a new column for the day
    df["Day"] = (df["Time"] // 24).astype(int)
    # Group by day and compute mean
    daily_df = df.groupby("Day").mean()

    # Create an array for mean values repeated for each hour
    mean_values_repeated = np.repeat(
        daily_df["Value"].values, df.groupby("Day").size().values
    )
    assert len(mean_values_repeated) == len(
        cumulative_dose_curve
    ), "mean_values_repeated and cumulative_dose_curve should have the same length"

    # Add label to the cumulative dose curve and plot it on the second subplot
    ax2.plot(time_axis, cumulative_dose_curve, label="Cumulative dose", color="black")
    ax2.plot(
        time_axis, mean_values_repeated, label="Daily mean cumul dose", color="red"
    )

    # Add legends to the curves
    ax1.legend(all_med_names_plot)
    ax1.set(title="Dose curves for HUP {}".format(patient_id), ylabel="Normalized dose")

    ax2.legend()
    ax2.set(
        title="Cumulative dose for HUP {}".format(patient_id),
        xlabel="Time (hours)",
        ylabel="Normalized dose",
    )

    for seizure_time in seizure_times:
        seizure_time = int(seizure_time[0]/3600 + aligned_emu_start_time_hrs)
        ax1.axvline(x=seizure_time, color="red", linestyle="--")
        ax2.axvline(x=seizure_time, color="red", linestyle="--")
        ax3.axvline(x=seizure_time, color="red", linestyle="--")

    return daily_df, hourly_df, cumulative_dose_curve, time_axis, ax3, fig



def plot_all_patients(hourly_dose_all, hourly_sync_all, frequency_range_str):
    """
    Plot the dose vs synchrony for all patients

    Parameters
    ----------
    hourly_dose_all : list
        List of lists of hourly dose values for each patient
    hourly_sync_all : list
        List of lists of hourly synchrony values for each patient
    frequency_range_str : str
        String of the frequency range of interest

    Returns
    -------
    None
    """

    # Find the min and max of hourly_dose_all
    plt.figure(figsize=(10, 8))
    all_dose = []
    all_sync = []

    # Iterate over each patient's data
    for i in range(len(hourly_dose_all)):
        dose = hourly_dose_all[i]
        sync = hourly_sync_all[i]

        # Ensure that only corresponding pairs are plotted
        min_length = min(len(dose), len(sync))

        plt.scatter(dose[:min_length], sync[:min_length], label=f"Patient {i+1}")

        all_dose.extend(dose[:min_length])
        all_sync.extend(sync[:min_length])

    # Convert to numpy arrays for calculation
    all_dose = np.array(all_dose)
    all_sync = np.array(all_sync)

    # Perform linear regression on all data
    slope, intercept, r_value, p_value, _ = stats.linregress(all_dose, all_sync)

    # Create array of x values for regression line
    x_values = np.linspace(min(all_dose), max(all_dose), 100)

    # Plot regression line
    plt.plot(x_values, slope * x_values + intercept, label="Linear Regression")

    # Plot the p value and r value in the plot
    plt.text(
        0.05,
        0.95,
        f"p value: {round(p_value, 5)}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        color="green",
    )
    plt.text(
        0.05,
        0.9,
        f"r value: {round(r_value, 5)}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        color="green",
    )
    plt.text(
        0.05,
        0.85,
        f"slope: {round(slope, 5)}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        color="green",
    )

    plt.xlabel("Dose")
    plt.ylabel("Synchrony")
    plt.title(f"Dose vs Synchrony for all Patients {frequency_range_str}")
    plt.legend()
    plt.show()

    # save the figure
    plt.savefig(
        f"./results/{frequency_range_str}/all_dose_vs_synchrony_{patient_id}.png"
    )


def linear_regression_and_plot(x, y, patient_id, frequency_range_str, frequency, type):

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x, y
    )
    print(
        f"slope: {slope}, intercept: {intercept}, r_value: {r_value}, p_value: {p_value}, std_err: {std_err}"
    )

    # Plot the linear regression model
    print("Plotting linear regression model...")
    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (10, 10)
    ax.plot(
        x,
        y,
        "o",
        label="original data",
    )
    ax.plot(
        x,
        intercept + slope * x,
        "r",
        label="fitted line",
    )
    ax.set_title(f"Dose vs synchrony for HUP {patient_id} {frequency_range_str}")
    ax.set_xlabel("Dose")
    ax.set_ylabel("R")
    ax.text(
        0.05,
        0.95,
        f"p value: {round(p_value, 5)}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        color="green",
    )
    ax.text(
        0.05,
        0.9,
        f"r value: {round(r_value, 5)}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        color="green",
    )
    ax.text(
        0.05,
        0.85,
        f"slope:   {round(slope, 5)}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        color="green",
    )

    # save the figure
    fig.savefig(
        f"./results/{frequency_range_str}/{patient_id}_{type}_dose_vs_synchrony_{frequency[0]}_{frequency[1]}.png"
    )