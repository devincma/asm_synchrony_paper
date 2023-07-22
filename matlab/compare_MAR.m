close all;clear;

% addpath('/mnt/leif/littlab/users/nghosn3/Pioneer/spikes-AED/aed_dose_modeling/figures_code')
% addpath('/mnt/leif/littlab/users/nghosn3/Pioneer/DATA');
% addpath('/mnt/leif/littlab/users/nghosn3/Pioneer/spikes-AED/aed_dose_modeling')
 addpath('/Volumes/USERS/nghosn3/Pioneer/spikes-AED/aed_dose_modeling/figures_code')
 addpath('/Volumes/USERS/nghosn3/Pioneer/DATA');
 addpath('/Volumes/USERS/nghosn3/Pioneer/spikes-AED/aed_dose_modeling')
      

% get asm data 
cohort_info = readtable('HUP_implant_dates.xlsx');
ptIDs = cohort_info.ptID;
weights = cohort_info.weight_kg;

load('MAR_032122.mat');
all_meds_asm = all_meds;

% get nlp data
all_meds_nlp = readtable('MedGivenCommonMRNs_DI.csv');

pt_inds = all_meds_nlp.ptID== 119;
temp_meds=all_meds_nlp(pt_inds,:);


ptIDs =unique(all_meds_nlp.ptID);

% find the HUP ID, if exists
hupids = nan(length(ptIDs),1);
for i=0:ptIDs(end)
    hup_ind = find(i==all_meds_nlp.ptID);
    if ~isempty(hup_ind)
    hupids(i+1) = all_meds_nlp.HUP_ID(hup_ind(1));
    else 
        hupids(i+1) = NaN;
    end
    
end

ptID_asm ='HUP141';
ptID_nlp = 79;

[med_names,meds,explant_date] = parse_MAR(ptID_asm,all_meds_asm);
[med_names_nlp,meds_nlp,explant_date_nlp,implant_date_nlp] = parse_MAR_nlp(ptID_nlp,all_meds_nlp);

%alcohol adinistrations:
alc_inds = contains(cohort_info.DISPLAY_NAME,'vodka') |  contains(cohort_info.DISPLAY_NAME,'wine');
alc_types = unique(cohort_info.DISPLAY_NAME(alc_inds));
alc_pts =  unique(cohort_info.ptID(alc_inds)); %69 patients/615



