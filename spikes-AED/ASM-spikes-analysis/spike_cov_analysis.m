%% first results for spike comparisons


close all;clear;
nina_path = '/Volumes/USERS/nghosn3/Pioneer';
addpath([nina_path '/DATA'])
addpath([nina_path '/spikes-AED/aed_dose_modeling/figures_code'])
addpath([nina_path '/spikes-AED/aed_dose_modeling'])

devin_path = '/Volumes/USERS/devinma/Pioneer';
addpath([devin_path '/spikes-AED/ASM-spikes-analysis'])

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
[all_pts_drug_samp, all_pts_tvec] = get_samp_asm_spikes(meds_fname,spikes_fname,all_ieeg_offset, all_dose_curves, all_tHr,ptIDs);
%%
% calculate the duration of data across all patients before the first
% seizure
sz_info = readtable('Erin_szs_times.xlsx','VariableNamingRule','preserve');

ind_first_seizure = zeros(length(ptIDs),1);
avg_rate_before_sz = zeros(length(ptIDs),1);
avg_rate_after_sz = zeros(length(ptIDs),1);
pre_ictal_rate = zeros(length(ptIDs),1);
post_ictal_rate = zeros(length(ptIDs),1);
interictal_rate = zeros(length(ptIDs),1);

for ipt =1:length(ptIDs)
    % get seizure times in ieeg and convert to idx of spike rate
    ptID = ['HUP' num2str(ptIDs(ipt))];
    spikes = all_spike_rate{ipt}';
    spikes = spikes./max(spikes); %normalize to compare across patients

    [med_names,meds,explant_date,starts_eeg,starts_emu] = parse_MAR(ptID,all_meds);
    [taper_info,any_taper] = get_taper_info(meds, med_names);


    pt_inds = find(contains(sz_info.Patient,ptID));
    pt_szData = sz_info(pt_inds,:);
    seizure_times = pt_szData.start;
    % index of seizure occurance, use data before
    first_ind = round(seizure_times(1) ./ (60*10));
    ind_first_seizure(ipt) = first_ind; % seconds / 60sec/min /10min/point

    % split the spikes, and remove data around the seizures - 1hr before
    % and 1 hour after
    two_hr = 6*2; % 10 min * 12 = 2hours of time
    before_spikes = spikes(1:(first_ind-two_hr));

    rm_inds = [];
    preictal_inds = [];
    postictal_inds =[];
    seizure_inds = round(seizure_times(2:end,1) ./ (60*10));

    for n = 1:length(seizure_inds)
        if seizure_inds(n)>two_hr
            rm_inds = [rm_inds seizure_inds(n)-two_hr:seizure_inds(n)+two_hr];
            preictal_inds = [preictal_inds seizure_inds(n)-two_hr:seizure_inds(n)];
            postictal_inds = [postictal_inds seizure_inds(n):seizure_inds(n)+two_hr];
        end
    end

    after_spikes = spikes;
    rm_inds(rm_inds > length(after_spikes))=[];
    postictal_inds(postictal_inds > length(spikes))=[];
    after_spikes(rm_inds)=[];

    avg_rate_before_sz(ipt) = mean(before_spikes);
    avg_rate_after_sz(ipt) =  mean(after_spikes);

    pre_ictal_rate(ipt) = mean(spikes(preictal_inds));
    post_ictal_rate(ipt) = mean(spikes(postictal_inds));
    %     interictal_rate(ipt) = ;


end

%% some figures;

% may want to exclude patients that haqd very early seizures
ex_pts = ~(ind_first_seizure < 6*6); % first seizure occured in first 6/12 hours


plot(avg_rate_before_sz(ex_pts),avg_rate_after_sz(ex_pts),'.b','markersize',20);
hold on;
plot(linspace(0,0.3,100),linspace(0,0.3,100),'--k','linewidth',3)
ylabel('norm spike rate after first sz','fontsize',14);
xlabel('norm spike rate before first sz','fontsize',14);
limits = [0 0.3];
ylim(limits); xlim(limits); axis square;
title('inter-ictal spike rate before vs after first seizure','fontsize',16)

% compare pre and post ictal spike rate across patients (aggregated across seizures?)
[h,p] = ttest(pre_ictal_rate,post_ictal_rate); % post ictal spike rate is higher??

%% spikes beginning and end of emu stay
early_late_emu_rate = zeros(length(ptIDs),2);
for ipt = 1:length(all_spike_rate)
    spikes = all_spike_rate{ipt}; spikes = spikes./max(spikes);
    early_late_emu_rate(ipt,:) = [mean(spikes(1:round(length(spikes)./5))) mean(spikes(round(length(spikes)./5)+1:end))];
end

% is the spike rate higher in the second half of the emu stay?
[h,p] = ttest(early_late_emu_rate(:,1),early_late_emu_rate(:,2)); % H = 0; no, the spike rate is not greater at the end

%% spike rate in low vs high ASM load states
high_asm_spikes = zeros(length(ptIDs),1);
low_asm_spikes = zeros(length(ptIDs),1);
med_asm_spikes = zeros(length(ptIDs),1);

for ipt = 1:length(ptIDs)

    ptID = ['HUP' num2str(ptIDs(ipt))];
    spikes = all_spike_rate{ipt}';
    spikes = (spikes)./max(spikes); %normalize to compare across patients
    asm_load = mean(all_pts_drug_samp{ipt});

    high_asm_inds = asm_load >= prctile(asm_load,50);
    low_asm_inds = asm_load <= prctile(asm_load,50);
    %med_asm_inds = asm_load > prctile(asm_load,25) &  asm_load < prctile(asm_load,75);


    high_asm_spikes(ipt) = mean(spikes(high_asm_inds));
    low_asm_spikes(ipt) = mean(spikes(low_asm_inds));
    %med_asm_spikes(ipt) = mean(spikes(med_asm_inds));

end

[p,tbl,stats] = kruskalwallis([low_asm_spikes high_asm_spikes],[{'low'} {'high'}]);

%%
% test
plot_spec = 1;

for ipt =77%:length(ptIDs)
    
    ptID = ['HUP' num2str(ptIDs(ipt))];
    asm_load = all_pts_drug_samp{ipt};
    spikes = all_spike_rate{ipt}; spikes = spikes./max(spikes);

    % split the asm load and spikes to before and after taper (no more
    % drug) - manually selected
    ind1 = 305; 
    ind2 = 817;
    asm_before = asm_load(:,1:ind1);
    asm_after = asm_load(:,ind1+1:ind2);
    
    spikes_before = spikes(:,1:ind1);
    spikes_after = spikes(:,ind1+1:ind2);

    med_names = all_med_names{ipt};
    plot_dim = length(med_names)+1;

    window =144;
    hrs =[1 2:48];
    f = (1./hrs) ./ 3600; % to units of samps/1sec (Hz)
    fs= 1/600;

    % before 
    figure;
    for m = 1:length(med_names) 
        subplot(plot_dim,1,m)
        %x=asm_load(m,:);
        x = asm_before(m,:);
        [pxx,f]=periodogram(x,window*ones(1,length(x)),f,fs,'power');
        %convert f to cycle/hours and un-normalize it
        fplot = (f .* (3600 .* (hrs.^2))) ;
        plot(fplot,(pxx));
        xlabel('period length (Hrs)')
        ylabel('dB')
        title(med_names{m})
        hold on;
    end
 
    
    subplot(plot_dim+1,1,plot_dim+1)
    %x=spikes;
    x = spikes_before;
    [pxx,f]=periodogram(x,window*ones(1,length(x)),f,fs,'power');
    %convert f to cycle/hours and un-normalize it
    fplot =( f .* (3600 .* (hrs.^2))) ;
    plot(fplot,(pxx));
    xlabel('frequency (Hrs)')
    ylabel('dB')
    title([ ptID ' spikes (normalized); before taper'])
    hold on;

    % after
    figure;
    for m = 1:length(med_names) 
        subplot(plot_dim,1,m)
        %x=asm_load(m,:);
        x = asm_after(m,:);
        [pxx,f]=periodogram(x,window*ones(1,length(x)),f,fs,'power');
        %convert f to cycle/hours and un-normalize it
        fplot = (f .* (3600 .* (hrs.^2))) ;
        plot(fplot,(pxx));
        xlabel('frequency (Hrs)')
        ylabel('dB')
        title(med_names{m})
        hold on;
    end
 
    
    subplot(plot_dim+1,1,plot_dim+1)
    %x=spikes;
    x = spikes_after;
    [pxx,f]=periodogram(x,window*ones(1,length(x)),f,fs,'power');
    %convert f to cycle/hours and un-normalize it
    fplot =( f .* (3600 .* (hrs.^2))) ;
    plot(fplot,(pxx));
    xlabel('period length (Hrs)')
    ylabel('dB')
    title([ ptID ' spikes (normalized); after taper (no meds)'])
    hold on;

end

function waterplot(s,f,t)
% Waterfall plot of spectrogram
waterfall(f,t,abs(s)'.^2)
set(gca,XDir="reverse",View=[30 50])
xlabel("Frequency (Hz)")
ylabel("Time (s)")
end




