%% Get the results spike detection from 5 min block every 30min (10-22-21: every 10min)
% we want to get the spike rate relative to the AED administration and
% seizure onset times

% get and save spike rate. make into function?
close all;clear;
curr_path = pwd;
addpath(curr_path)
addpath([curr_path '/DATA'])
addpath([curr_path '/spikes-AED'])


% Get which patients
cohort_info = readtable('HUP_implant_dates.xlsx');
ptIDs = cohort_info.ptID;

% save file?
save_file = 1;

all_spike_rate = cell(1,length(ptIDs));
all_spike_times = cell(1,length(ptIDs));
file_inds = cell(1,length(ptIDs));
spike_file_dur = zeros(3,length(ptIDs));

for x = 1:length(ptIDs)
    
    ptID = ['HUP' num2str(ptIDs(x))];
    %load the spike data
    addpath('/Volumes/USERS/erinconr/projects/fc_toolbox/results/analysis/intermediate/');
    fname = [ptID '.mat'];
    try
        load(fname);
        % later can remove spikes that occur during seizures
        spike_rate = nansum(summ.spikes); %spikes across all channels!!
        
        spike_rate = spike_rate*10;
        all_spike_rate(x) = {spike_rate};
        all_spike_times(x) = {summ.times};
        file_inds(x) = {summ.file_index};
        
        file_ind = find(diff(summ.file_times) <0);
        if ~isempty(file_ind)
            spike_file_dur(:,x) =[file_ind]; %in blocks
        end
    catch
        ptIDs(x)=NaN;
    end
end

%% save file in Pioneer/DATA
if save_file
    dirr = '/Volumes/USERS/nghosn3/Pioneer/DATA/spikes_rates_021323.mat';
    save(dirr,'all_spike_rate','ptIDs','spike_file_dur','all_spike_times','file_inds');
    disp 'saved downsampled spikes: 10min blocks'
end
