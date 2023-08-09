from pioneer import Pioneer
import pandas as pd
import numpy as np


def pipeline(index, table, rootpath):
    ID = table.loc[index, "ieeg_fname"]
    start = table.loc[index, "ieeg_start"] * 1e6  # CHECK IF USEC
    end = table.loc[index, "ieeg_end"] * 1e6  # CHECK IF USEC
    rosalind = Pioneer(
        "wojemann",
        rootpath + "woj_ieeglogin.bin",
        ID,
        start,
        end,
        np.array(table.loc[index, "electrodes"]),
    )
    # rosalind = Pioneer('pattnaik',rootpath + 'pre-ictal-similarity/code/pat_ieeglogin.bin',ID,start,end,np.array(table.loc[index,'electrodes']))
    # get patient data
    rosalind.pull_data()
    # filter and remove channels with too much noise
    rosalind.preprocess()
    # detect spikes
    rosalind.detect_spikes()
    output = rosalind.spike_data

    total_spikes = len(output)
    if total_spikes == 0:
        unique_spikes = 0
        recruited_channels = 0
    else:
        unique_spikes = len(np.unique(output[:, 2]))
        recruited_channels = len(np.unique(output[:, 1]))

    return (unique_spikes, recruited_channels, total_spikes)


def main():
    rootpath = "/users/wojemann/"
    # table = pd.read_json(rootpath+'iEEG_processing/spike_info.json', orient='records')
    ##### SINGLE RECORDING DEBUGGING
    # print(pipeline(60,table,rootpath))
    unique_spikes = []
    recruited_channels = []
    total_spikes = []
    for i in [68]:
        # ID = table.loc[i,'ieeg_fname']
        ID = "HUP212_phaseII"
        # start = table.loc[i,'ieeg_start']*1e6 # CHECK IF USEC
        # end = table.loc[i,'ieeg_end']*1e6 # CHECK IF USEC
        # rosalind = Pioneer('wojemann',rootpath + 'woj_ieeglogin.bin',ID,start,end)
        rosalind = Pioneer(
            "wojemann", rootpath + "woj_ieeglogin.bin", ID
        )  # ,start,end,np.array(table.loc[i,'electrodes']))
        rosalind.pull_annotations()
        rosalind.filter_seizure_annotations()
        print(rosalind.seizure_annotations)
        # get patient data
        # rosalind.pull_data()
        rosalind.channel_labels = np.array(table.loc[i, "electrode_names"])
        # filter and remove channels with too much noise
        rosalind.preprocess("bipolar")
        # detect spikes
        rosalind.detect_spikes()
        output = rosalind.spike_data
        if len(output) > 0:
            total_spikes.append(len(output))
            unique_spikes.append(len(np.unique(output[:, 2])))
            recruited_channels.append(len(np.unique(output[:, 1])))
        else:
            total_spikes.append(0)
            unique_spikes.append(0)
            recruited_channels.append(0)
        print(total_spikes, unique_spikes, recruited_channels)
    #     table.at[i, "unique_spikes"] = unique_spikes[-1]
    #     table.at[i, "recruited_channels"] = recruited_channels[-1]
    #     table.at[i, "total_spikes"] = total_spikes[-1]
    ################################

    ##### PIPELINE FUNCTION IMPLEMENTATION
    # kwargs = [{'index': index, 'table': table, 'rootpath': rootpath} for index in table.index]

    # result = pqdm(kwargs, pipeline, n_jobs=5, desc='iEEG processing pipeline', argument_type='kwargs')

    # table['spike_results'] = result
    ######################################

    # table.to_csv(rootpath+'iEEG_processing/Akash_table_wcounts_fixed.csv')


if __name__ == "__main__":
    main()
