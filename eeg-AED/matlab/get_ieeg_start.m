function ieeg_start = get_ieeg_start(ipt)
    addpath(['/Volumes/USERS/nghosn3/Pioneer/DATA'])
    addpath(['/Volumes/USERS/nghosn3/Pioneer/spikes-AED/aed_dose_modeling/figures_code'])
    addpath(['/Volumes/USERS/nghosn3/Pioneer/spikes-AED/aed_dose_modeling'])
    cd('/Volumes/USERS/nghosn3/Pioneer/spikes-AED');
    tic

    %load spike rate - new from 2/13/23 (samp/10min)
    load('spikes_rates_021323.mat');
    inds = cellfun('isempty',all_spike_rate);

    cohort_info = readtable('HUP_implant_dates.xlsx');
    ptIDs = cohort_info.ptID;
    weights = cohort_info.weight_kg;
    load('MAR_032122.mat')

    [all_dose_curves,all_tHr,ptIDs,all_med_names,all_ieeg_offset,max_dur,emu_dur] = get_aed_curve_kg(ptIDs,weights);
    %...

    % patient medication administration
    ptID =  ['HUP' num2str(ptIDs(ipt))];

    [~,~,~,starts_eeg,starts_emu] = parse_MAR(ptID,all_meds);

    offsets = all_ieeg_offset{2,ipt};
    ieeg_offset_datasets = all_ieeg_offset{1,ipt};
    spike_rate=all_spike_rate{ipt}; %calculated spike rate for patient in list
    %spike_rate=log10(all_spike_rate{ipt}+1);
    time = all_spike_times{ipt};

    % align ieeg times for each file with emu medication times
    offset_vec = file_inds{ipt};
    for i = unique(offset_vec)'
        time_inds = (offset_vec==i);
        time(time_inds) = time(time_inds) - starts_eeg(i) + starts_emu(i);
    end 
    %time(offset_vec==1) = time(offset_vec==1) + offsets(1); %shift for t=0 to be start of emu stay, not start of ieeg recording
    time =time +offsets(1);

    % Saving the first element of time./3600 to return
    ieeg_start = time(1)./3600;

    % Rest of the code...
    %...
    cd('/Volumes/USERS/devinma/eeg-AED/matlab')
end
