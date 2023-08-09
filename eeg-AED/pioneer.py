from get_iEEG_data import *
from IES_detector import *
from preprocessing import *
from IES_helper_functions import *
from ieeg.auth import Session
from scipy.io import savemat
import os
from os.path import exists as ospe
from os.path import join as ospj


class Pioneer:
    """
    Pioneer class for EEG processing based on the iEEG remote database and associated api.
    ToDo:
        allow for individual preprocessing steps within class
        interaction with (seizure) annotations
        loading data into same format from other sources (e.g. csv,imaging?)
        BCT integration
    """

    def __init__(
        self, username, passpath, ID, start=0, end=0, channel_labels=[], savepath=None
    ) -> None:
        self.ID = ID
        self.username = username
        self.passpath = passpath
        self.savepath = savepath
        self.start_time = start
        self.end_time = end
        self.duration = end - start
        self.channel_labels = channel_labels
        self.fs = 0
        self.data = None
        self.spike_data = None

        if not isinstance(channel_labels, np.ndarray):
            channel_labels = np.array(channel_labels)
        # figure out plan for this
        self.fig_path = "/users/wojemann/iEEG_processing/figures/"

        with open(passpath, "r") as f:
            s = Session(username, f.read())
        dataset = s.open_dataset(ID)
        # Either manually set channel labels or grab them from iEEG
        self.all_labels = np.array(dataset.get_channel_labels())
        if any(channel_labels):
            self.manual_channels = True
            self.channel_labels = channel_labels
        else:
            self.manual_channels = False
            label_idxs = electrode_selection(self.all_labels)
            self.channel_labels = self.all_labels[label_idxs]

    def pull_data(self, start=None, end=None):
        if start:
            start_time = start
            end_time = end
        elif self.start_time:
            start_time = self.start_time
            end_time = self.end_time
        else:
            raise ("No start and end time specified, please give boundaries for clip")
        data, fs = get_iEEG_data(
            self.username,
            self.passpath,
            self.ID,
            start_time,
            end_time,
            self.channel_labels,
        )
        self.data = data
        self.fs = fs

    def apply_function(self, function, args=None, kwargs=None):
        dataframe = pd.DataFrame(self.data, columns=self.channel_labels)
        new_df = dataframe.apply(function, 0, args=args, kwargs=kwargs)
        self.data = new_df.to_numpy()

    def pipeline(self):
        """
        Allow users to pass through a list of functions or strings of functions to specify the order of processes
        """
        pass

    def detect_spikes(self, plotting=False, return_count=False):
        """
        Function to apply ies_detector function to pioneer class patient.
        See ies_detector documentation for more details on implementation and inputs.
        Outputs of the function are stored in self.spike_data and self.spike_count where
        self.spike_data is a nx3 NDArray where the first column is the spike time relative
        to the start of the window, the second is the channel the spike occured on, and the
        third denotes which train the spike was part of.

        inputs:
        plotting    -   (bool) flag to save plots generated from the spike detection algorithm
        return_count-   (bool) flag to return the output of the spike detector

        returns:
        (optional)
        spike_count -   (NPArray)
        """
        self.spike_data = ies_detector(
            self.data.to_numpy(),
            self.fs,
            plot_sig=plotting,
            labels=self.referenced_labels,
            fig_path=self.fig_path,
        )
        if len(self.spike_data) == 0:
            self.spike_count = 0
        else:
            self.spike_count = len(np.unique(self.spike_data[:, 2]))
        if return_count:
            return self.spike_data

    def preprocess(self, ref_type="car"):
        """
        Function to manually set the channels for analysis. Applies preprocessing function to
        all channels of the dataset. Preprocessing function includes 60Hz notch filtering and
        1-120Hz bandpass filtering.

        Inputs:
        ref_type    -   (String) 'car' or 'bipolar' implementing either a common average reference
                                 or bipolar montage.
        Returns:
        None
        """
        filt_data, filt_labels = preprocess(
            self.data.to_numpy(),
            self.fs,
            self.channel_labels,
            self.manual_channels,
            ref_type,
        )
        self.data = pd.DataFrame(filt_data, index=self.data.index, columns=filt_labels)
        self.referenced_labels = filt_labels

    def filter_bad_channels(self):
        channel_mask, details = detect_bad_channels(
            self.data.to_numpy(), self.fs, self.channel_labels
        )
        self.data = self.data.iloc[:, channel_mask]
        self.channel_labels = self.channel_labels[channel_mask]

        return details

    def set_channels(self, channels):
        """
        Function to manually set the channels for analysis
        Inputs:
        channels    -   (nparray) list of channel names or indices to use for analysis
        Returns:
        None
        """
        # Add option for creating a new montage
        self.manual_channels = True
        self.channel_labels = channels

    def pull_annotations(self, return_annots=False, save_path=None):
        """
        Function to pull the annotations of the pioneer class patient
        """
        with open(self.passpath, "r") as f:
            s = Session(self.username, f.read())
        dataset = s.open_dataset(self.ID)
        all_layers = dataset.get_annotation_layers()
        layer_name = list(all_layers.keys())[0]
        expected_count = all_layers[layer_name]

        actual_count = 0
        max_results = None if expected_count < 100 else 100
        call_number = 0
        all_annotations_li = []
        while actual_count < expected_count:
            annotations = dataset.get_annotations(
                layer_name, first_result=actual_count, max_results=max_results
            )
            call_number += 1
            actual_count += len(annotations)

            for annotation in annotations:
                first = pd.to_datetime(annotation.start_time_offset_usec, unit="us")
                duration = (
                    pd.to_datetime(annotation.end_time_offset_usec, unit="us") - first
                )
                description = annotation.description
                all_annotations_li.append([first, duration, description])

            first = annotations[0].start_time_offset_usec
            last = annotations[-1].end_time_offset_usec
            description = annotations[0].description

            print(
                "got",
                len(annotations),
                "annotations on call #",
                call_number,
                "covering",
                first,
                "usec to",
                last,
                "usec",
            )
        all_annotations_df = pd.DataFrame(all_annotations_li)
        all_annotations_df.columns = ["Time", "Duration", "Description"]
        all_annotations_df.set_index("Time", inplace=True)
        self.annotations = all_annotations_df
        if save_path:
            save_path = ospj(
                self.savepath, self.ID, "ieeg_features", f"{self.ID}_annotations.h5"
            )
            if not ospe(ospj(self.savepath, self.ID, "ieeg_features")):
                os.makedirs(ospj(self.savepath, self.ID, "ieeg_features"))
            all_annotations_df.to_hdf(save_path, key="annotations", mode="w")
        if return_annots:
            return all_annotations_df

    def filter_seizure_annotations(self):
        keywords = [
            "seizure",
            "sz",
            "event",
            "jerks",
            "auras",
            "episodes",
            "myoclon",
            "spell",
            "EEC",
            "UEO",
            "onset",
            "offset",
            "disrupt",
            "cortical stim",
            "ictal",
        ]
        keywords_regex = "|".join(keywords)
        # for making it case insenstive
        keywords_regex = "^(?i)" + keywords_regex
        mask = self.annotations["Description"].str.contains(keywords_regex)
        mask[mask.isna()] = False

        filtered_annotations = self.annotations[mask]

        filter_pct = 1 - len(filtered_annotations) / len(self.annotations)
        print("Filtered {:10.2f}% of all annotations".format(filter_pct * 10))

        t_sec = [i.timestamp() for i in filtered_annotations.index]
        filtered_annotations["IEEG Time"] = t_sec
        self.seizure_annotations = filtered_annotations

    def save2mat(self, filepath="", object="", patient_info=False):
        """
        Function to save generated data stored in the class as a matlab file
        """
        if patient_info:
            output = {"ID": self.ID, "fs": self.fs, object: eval(object)}
            savemat(filepath, output)
        else:
            savemat(filepath, {object: eval(object)})

    def bandpower(self):
        pass

    def plot_signal(self):
        """
        Function for plotting the data and any associated, calculated metadata contained in the
        object (e.g. spike locations, annotations, etc.)
        """
        pass

    def plot_eeg(self, figsize=(24, 8), savefig=False):
        """
        This functino plots EEG data with tailored offets for each channel to remove
        signal overlap in plotting

        Inputs:
        data    - nd.array - 2D array containing EEG data as a numpy array with samples
                    as rows and channels as columns
        fs      - int - sampling frequency of the EEG data
        labels  - Str collection - labels of the eeg channels [default = None]
        figsize - 2 x 1 int collection - dimensions for the plot [default = (24,8)]

        Returns:
        None
        """

        fig, ax = plt.subplots(figsize=(24, 8))  ### CHANGE THIS FOR MORE CHANNELS
        channel_offsets = np.insert(
            np.cumsum(
                np.abs(np.min(self.data.T[:, 1:], axis=0))
                + np.max(self.data.T[:, :-1], axis=0)
            ),
            0,
            0,
        )
        temp = self.data.T + channel_offsets
        time = np.arange(len(temp)) / fs
        ax.plot(time, temp, "k", linewidth=0.2)
        ax.set_yticks(channel_offsets)
        ax.set_yticklabels(self.channel_labels)
        if savefig:
            fig.savefig(self.fig_path + "all_spikes.png")
        plt.close(fig)
