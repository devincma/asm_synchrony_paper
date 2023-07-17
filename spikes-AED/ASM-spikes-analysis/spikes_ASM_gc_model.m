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
        drug=drug./max(drug, [], "omitmissing"); %normalize each drug curve
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
%% time series analysis - VAR model for granger causality

all_gc_results = nan(1,length(ptIDs));

for ipt = 1:2%length(ptIDs)
    % test for stationarity and make data stationary - spikes are generally
    % stationary but asm load is not
    asm_load = mean(all_pts_drug_samp{ipt},1)'; % averaged ASM load across medications
    spikes = all_spike_rate{ipt}';

    % make the asm load stationary by takking the differende between points
    % (derivative)
    asm_load_diff = [diff(asm_load); 0];% add a point at the beginning since it will be one less

    % need to normalize the spike rates
    spikes_norm = spikes ./max(spikes);


    % remove the time points with zeros spikes?
    % zero_inds = spikes==0;
    % asm_load(zero_inds)=[];
    % spikes(zero_inds)=[];


    % store pre-processed data into a table for model fitting and selection -
    % do this for each patient - start with example 1

    tbl = table(asm_load_diff,spikes_norm);
    T = size(tbl,1);

    % now need to find the optimal number of lags
    numseries = 2;
    numlags = (1:20)'; %(lags are in 10min blocks, so try 1:5 hours in increments of 1 hr)
    nummdls = numel(numlags);

    % Partition time base.
    maxp = max(numlags) * 1; % Maximum number of required presample responses - try making the pre estimate time base longer
    idxpre = 1:maxp;
    idxest = (maxp + 1):T;

    % Preallocation
    EstMdl(nummdls) = varm(numseries,0);
    aic = zeros(nummdls,1);

    % Fit VAR models to data.
    Y0 = tbl{idxpre,:}; % Presample
    Y = tbl{idxest,:};  % Estimation sample
    for j = 1:numel(numlags)
        Mdl = varm(numseries,numlags(j));
        Mdl.SeriesNames = string(tbl.Properties.VariableNames);
        EstMdl(j) = estimate(Mdl,Y,'Y0',Y0);
        results = summarize(EstMdl(j));
        aic(j) = results.AIC;
    end

    [~,bestidx] = min(aic);
    p = numlags(bestidx) %if p=6 then the best number of lags is 6 which is the same as one hour

    % now we take the best model and do a granger causality test

    BestMdl = EstMdl(bestidx);
    [h,Summary] = gctest(BestMdl,'Display',false); % gc test shows that the asm load granger causes the spike rate???

    all_gc_results(ipt) = h(2); % we want the result of testing the hyp for the affect of asm_load on spikes rate 

end


% can we conduct a test across all patients to see if this is generally
% true?

%% SARIMAX - seasonal arima model with exogenous variable

