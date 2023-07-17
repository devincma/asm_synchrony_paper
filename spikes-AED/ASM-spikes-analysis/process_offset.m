close all;clear;
nina_path = '/Volumes/USERS/nghosn3/Pioneer';
addpath([nina_path '/DATA'])
addpath([nina_path '/spikes-AED/aed_dose_modeling/figures_code'])
addpath([nina_path '/spikes-AED/aed_dose_modeling'])

tic

%load spike rate and med data - new from 2/13/23 (samp/10min)
spikes_fname = 'spikes_rates_021323.mat';
load(spikes_fname);

cohort_info = readtable('HUP_implant_dates.xlsx');

ptIDs = cohort_info.ptID;
weights = cohort_info.weight_kg;

meds_fname = 'MAR_032122.mat';
load(meds_fname);

[all_dose_curves,all_tHr,ptIDs,all_med_names,all_ieeg_offset,max_dur,emu_dur] = get_aed_curve_kg(ptIDs,weights);


row1 = all_ieeg_offset(1, :);
row1_og = all_ieeg_offset(1, :);
row2 = all_ieeg_offset(2, :);
row2_og = all_ieeg_offset(2, :);
row3 = all_ieeg_offset(3, :);
row3_og = all_ieeg_offset(3, :);

for i = 1:length(row2)
    local_cell = row2_og{i};
    n = size(local_cell,1);
    if n > 0
        for j = 1:n
            row2{j,i} = string(local_cell(j,1));
        end
    end
end