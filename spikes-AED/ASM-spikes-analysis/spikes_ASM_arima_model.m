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

%%
[all_dose_curves,all_tHr,ptIDs,all_med_names,all_ieeg_offset,max_dur,emu_dur] = get_aed_curve_kg(ptIDs,weights);
[all_pts_drug_samp, all_pts_tvec] = get_samp_asm_spikes(meds_fname,spikes_fname,all_ieeg_offset, all_dose_curves, all_tHr,ptIDs);

%% time series analysis - VAR model for granger causality, and cross correlation

all_gc_results = nan(1,length(ptIDs));
xcorr_resuls = table();

row_ind_start =1;
for ipt = 1%:length(ptIDs)
    ptID = ['HUP' num2str(ptIDs(ipt))];
    % test for stationarity and make data stationary - spikes are generally
    % stationary but asm load is not
    asm_load = all_pts_drug_samp{ipt}; % averaged ASM load across medications
    spikes = all_spike_rate{ipt}';
    
    [med_names,meds,~,~,~] = parse_MAR(ptID,all_meds);
    [taper_info,any_taper] = get_taper_info(meds, med_names);
    % make the asm load stationary by taking the difference between points (derivative)
    %asm_load_diff = [diff(asm_load); 0];% add a point at the beginning since it will be one less

    % need to normalize the spike rates
    spikes_norm = spikes ./max(spikes);
    
    max_lags = 200;
    r_maxes = zeros(1,height(asm_load));
    lag_maxes =  zeros(1,height(asm_load));
    med_tapered = zeros(1,height(asm_load));

    for l = 1:height(asm_load)
        [r,lags] = xcorr(spikes_norm,asm_load(l,:),'normalized');
        [maxr, ind] = max(r);
        r_maxes(l) = maxr;
        lag_maxes(l) = lags(ind);
        ind = contains({taper_info.med_name},med_names{l});
        if sum(ind)>0
            med_tapered(l) = taper_info(ind).tapered;
        end
    end

    xcorr_results.r_max(row_ind_start:row_ind_start+height(asm_load)-1) = r_maxes';
    xcorr_results.lags(row_ind_start:row_ind_start+height(asm_load)-1) = lag_maxes';
    xcorr_results.med_name(row_ind_start:row_ind_start+height(asm_load)-1) = med_names;
    xcorr_results.ptID(row_ind_start:row_ind_start+height(asm_load)-1) = ipt*ones(length(med_names),1);
    xcorr_results.tapered(row_ind_start:row_ind_start+height(asm_load)-1) = med_tapered';
    
    row_ind_start = row_ind_start+height(asm_load);
    % remove the time points with zeros spikes?
    % zero_inds = spikes==0;
    % asm_load(zero_inds)=[];
    % spikes(zero_inds)=[];

    % store pre-processed data into a table for model fitting and selection -
    % do this for each patient - start with example 1

    T = length(asm_load);

    p = 6; %max lag is p=6 (1hr)
    D = 1; %nonseasonality difference lags
    q = 0; %moving average terms - 0 for now

    sar_lags = 72; %144 lags is 24 hours - corresponds to sleep wake cycles 
    P = sar_lags+p+1;
%     % Partition time base.
%     idxpre = 1:P;
%     idxest = (P + 1):T;
    
    % specify the SARIMAX model - ARIMA w/ periodicity and exogenous
    % variables
    Mdl = arima('ARLags',1:p,'D',D,'Seasonality',sar_lags);
    %    Mdl = arima('ARLags',1:p,'D',D);


    % Fit ARIMA model to data.
    split =round(.75*length(spikes_norm));
    Y0 = spikes_norm(1:split); % Presample
    Y = spikes_norm(split+1:end);  % Estimation sample
    EstMdl_asm = estimate(Mdl,Y,'Y0',Y0,'X',asm_load(1:split)');
    EstMdl_spikes = estimate(Mdl,Y,'Y0',Y0);
    
    % forcast spike rate - these should be the same for model with and without ASM load
    forcast_period = P;% forcast P spike rate - 30hrs
    idxpre = 1:EstMdl_asm.P;
    idxest = (EstMdl_asm.P + 1):(length(spikes_norm)-forcast_period);
    
    YF0 = spikes_norm(idxest(end-P+1:end)); % allocate the presample for the forcast estimation
    YF = forecast(EstMdl_asm,forcast_period,YF0);
    YF_null = forecast(EstMdl_spikes,forcast_period,YF0);
   
    
    figure;
    plot(spikes_norm,'b'); hold on;
    forecast_time = (length(spikes_norm)-forcast_period+1):length(spikes_norm);
    plot(forecast_time,YF,'r');
    plot(forecast_time,YF_null,'g');
    legend('actual','forecasted with asm load','forecasted without asm load')
    
    figure; % plot residuals 
    histogram(spikes_norm(forecast_time)-YF); hold on;
    histogram(spikes_norm(forecast_time)-YF_null); hold on;

    % need a way to test if the exogenous variable terms make the model
    % better - esitmagte with/without ASM load and then forcast and check
    % residuals 
end


% can we conduct a test across all patients to see if this is generally
% true?

%% SARIMAX - seasonal arima model with exogenous variable

