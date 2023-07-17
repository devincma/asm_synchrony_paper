import pandas as pd


def get_seizure_times(hup_id):
    # hup_id is in the format of "HUP190"
    all_seizures = pd.read_excel(
        "../../Data/Erin_szs_times.xlsx", sheet_name="Erin_szs_times"
    )

    dataset = all_seizures["IEEGname"].apply(
        lambda x: str(x).split("_") if pd.notnull(x) else ["one file"]
    )

    dataset_name = [item[-1] if "D" in item[-1] else "one file" for item in dataset]
    all_seizures["Dataset"] = dataset_name

    # get the seizure times
    pt_inds = all_seizures["Patient"].str.contains(hup_id, na=False)
    pt_seizure_info = all_seizures[pt_inds]

    seizure_times = pt_seizure_info[["start", "end"]]
    seizure_dataset = pt_seizure_info["Dataset"]

    return seizure_times, seizure_dataset
