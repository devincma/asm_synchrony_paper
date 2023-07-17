%% sample the medication curve at spike times to run time bin and correlation analyses

close all;clear;
nina_path = '/Volumes/USERS/nghosn3/Pioneer';
addpath([nina_path '/DATA'])
addpath([nina_path '/spikes-AED/aed_dose_modeling/figures_code'])
addpath([nina_path '/spikes-AED/aed_dose_modeling'])

tic


%load spike rate - new from 2/13/23 (samp/10min)
load('spikes_rates_021323.mat');
inds = cellfun('isempty',all_spike_rate);

cohort_info = readtable('HUP_implant_dates.xlsx');
ptIDs = cohort_info.ptID;
weights = cohort_info.weight_kg;
load('MAR_032122.mat')

[all_dose_curves,all_tHr,ptIDs,all_med_names,all_ieeg_offset,max_dur,emu_dur] = get_aed_curve_kg(ptIDs,weights);
%%
all_pts_drug_samp = cell(1,length(ptIDs));
all_pts_tvec = cell(1,length(ptIDs));

for ipt = 1:length(ptIDs)
    
    ptID =  ['HUP' num2str(ptIDs(ipt))];
    [med_names,meds,explant_date,starts_eeg,starts_emu] = parse_MAR(ptID,all_meds);
    
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
    time = (time + offsets(1))/3600; %shift for t=0 to be start of emu stay, not start of ieeg recording. convert to hours
    
    %sample the med curves to be time aligned with the spikes 
    pt_drug_curves = all_dose_curves{ipt};
    pt_tHr = all_tHr{ipt};
    drugs_samp =zeros(length(med_names),length(spike_rate)); %450 hours of EMU stay in minutes
    for i =1:length(med_names)
        drug=pt_drug_curves{i};
        drug=drug./nanmax(drug); %normalize each drug curve
        drug_samp = zeros(1,length(spike_rate));
        if ~isempty(drug)
            for t = 1:length(time)
                if time(t)-pt_tHr{i}(1) >0
                    [~,t_ind] = min(abs(time(t)-pt_tHr{i})); % find closest time point 
                    drug_samp(t) = drug(t_ind);
                end
            end
            drugs_samp(i,:)=drug_samp;
        end
    end
    assert(length(time) == length(drugs_samp));
    all_pts_drug_samp(ipt) = {drugs_samp}; 
    all_pts_tvec{ipt} = {time};
    
    
end 

%% use drugs_samp and time and spike rate to run some analyses 
% spikes vs asm phase plot
for i =1:length(all_spike_rate)
    
    subplot(8,10,i)
    plot(mean(all_pts_drug_samp{i}),(all_spike_rate{i}+1),'.','markersize',3);
end 



%% time since last seizure

all_t_last_sz = cell(1,length(ptIDs));
for ipt = 1:length(ptIDs)
    
    ptID =  ['HUP' num2str(ptIDs(ipt))];
    [med_names,meds,explant_date,starts_eeg,starts_emu] = parse_MAR(ptID,all_meds);
    nbins = length(all_pts_drug_samp{ipt}); % spikes are sampled every 10mins
    
    offsets = all_ieeg_offset{2,ipt};
    ieeg_offset_datasets = all_ieeg_offset{1,ipt};
    [seizure_times,seizure_dataset] = get_seizure_times_from_sheet(ptID);
    if ~isempty(offsets)
        for j =1:height(seizure_times)
            % check which dataset the seizure is from, and add appropriate offset
            if isequal(seizure_dataset{j},'D01') || isequal(seizure_dataset{j},'one file')
                seizure_times(j,1)= (offsets(1)+(seizure_times(j,1)))./3600;
            else
                ind = str2double(seizure_dataset{j}(end));
                if 1 %sum(ind)>0
                    dataset_offset = offsets(1) - starts_eeg(ind) + starts_emu(ind);
                    seizure_times(j,1)= (seizure_times(j,1) + dataset_offset)./3600; %convert to hours
                end
            end
        end
    end
    
    % Get time since last seizure
    t_last_seizure = nan(1,nbins);
    seizure_inds = unique(round(seizure_times(:,1)));
    start = seizure_inds(1);
    ind=1;
    for j=start:seizure_inds(end)
        if ind<length(seizure_inds)
            if j>=seizure_inds(ind) && j < seizure_inds(ind+1)
                t_last_seizure(j)=j-seizure_inds(ind);
                
            else
                ind=ind+1;
                t_last_seizure(j)=j-seizure_inds(ind);
            end
        end
    end
    t_last_seizure(j+1:end) = (j+1:length(t_last_seizure))-seizure_inds(end);
    
    % assume last seizure happened 'first_sz' hours before emu stay and add that in
    % beginning
    nan_inds = find(isnan(t_last_seizure));
    first_sz = -24*2; 
    first_inds = nan_inds - first_sz;
    t_last_seizure(nan_inds)=first_inds;
    all_t_last_sz(ipt) = {t_last_seizure};
    
    
end

%% create feature matrix - model w/ response of spike rate to see coefficients - lme?
feat_names = [{'ptID'}, {'asm_load'}, {'t_last_sz'} {'log_spike_rate'}];
features = cell(4,length(ptIDs));
for i = 1:length(ptIDs)
    % add ptID
    features{1,i} = i*ones(1,length(all_pts_drug_samp{i}));
    % add asm load 
    features{2,i} = mean(all_pts_drug_samp{i});
    % add t_last_sz
    features{3,i} = all_t_last_sz{i};
    % add spike rate
    features{4,i} = all_spike_rate{i};
end 

drug_table = table();
for i =1:length(feat_names)
    feat_name = feat_names{i};
    this_feat = features(i,:);
    this_feat = horzcat(this_feat{:});
    drug_table.(feat_name)=this_feat';
end
%drug_table.ptID = categorical(drug_table.ptID);

%% fit linear mixed effects regression model 
modelspec = 'log_spike_rate~asm_load+t_last_sz+(1|ptID)';
mdl = fitglme(drug_table,modelspec); % 'Distribution','Poisson'

toc