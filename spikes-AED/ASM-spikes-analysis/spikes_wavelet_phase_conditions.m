%% spectral analysis of spike rate using wavelet transform

% Use Nina's path for all data
close all;clear;
nina_path = '/Volumes/USERS/nghosn3/Pioneer';
addpath([nina_path '/DATA'])
addpath([nina_path '/spikes-AED/aed_dose_modeling/figures_code'])
addpath([nina_path '/spikes-AED/aed_dose_modeling'])
addpath('/Volumes/users/nghosn3/tools/wavelet-coherence')

% Use Devin's path for his own code
devin_path = '/Volumes/USERS/devinma/Pioneer';
addpath([devin_path '/spikes-AED/ASM-spikes-analysis'])

%load spike rate and med data - new from 2/13/23 (samp/10min)
spikes_fname = 'spikes_rates_021323.mat';
load(spikes_fname);
% get the seizure and SOZ localization information 
soz_info = readtable('Erin_szs_times.xlsx','VariableNamingRule','preserve','Sheet','SOZ');

cohort_info = readtable('HUP_implant_dates.xlsx');
ptIDs = cohort_info.ptID;
weights = cohort_info.weight_kg;

meds_fname = 'MAR_032122.mat';
load(meds_fname);

[all_dose_curves,all_tHr,ptIDs,all_med_names,all_ieeg_offset,max_dur,emu_dur] = get_aed_curve_kg(ptIDs,weights);
[all_pts_drug_samp, all_pts_tvec] = get_samp_asm_spikes(meds_fname,spikes_fname,all_ieeg_offset, all_dose_curves, all_tHr,ptIDs);

%% find patients for spectral analysis: 1) taper and 'off' of at least one to all medications for >2days
pt_data_clips = table();
pt_data_clips.ptID = (ptIDs);
pt_data_clips.med_names = all_med_names';
pt_data_clips.inds = cell(length(ptIDs),1);

days=2; % threshold in days
taper_thresh = days*(24*60)./10; % 10min segments in 2 day period

for ipt =8%1:length(ptIDs)
    ptID = ['HUP' num2str(ptIDs(ipt))]
    asm_curves = all_pts_drug_samp{ipt};
    med_names = all_med_names{ipt};
    offsets = all_ieeg_offset{2,ipt};
    ieeg_offset_datasets = all_ieeg_offset{1,ipt}; 

    %find when each drug was stopped and find period of halt
    inds = nan(length(med_names),3);
    for n=1:length(med_names)
        if ~(contains(med_names{n},'lorazepam'))
        med_curve = asm_curves(n,:);
        [pks,locs] = findpeaks(med_curve);
        diffs = diff(locs);
        if length(diffs)>1
            locs(end+1) = length(med_curve);
            diffs = diff(locs);
        end
        last_admin = find(diffs >= taper_thresh);
        if ~isempty(last_admin)
            stop_med =locs(last_admin(1)); %ind+1 of last administration
            restart_med =locs(last_admin(1)+1)-1; % ind -1 of next peak after stopping

            inds(n,1) = 1;
            inds(n,2) = stop_med;
            inds(n,3) = restart_med;
        end
        end 
    end
    pt_data_clips.inds(ipt) = {inds};

    % add seizure times 
    [seizure_times] = get_seizure_times_from_sheet(ptID);
    
    sz_inds = zeros(length(seizure_times),1);
    for sz = 1:length(seizure_times)
        [~,sz_ind] = min(abs(all_spike_times{ipt}-seizure_times(sz,1)));
        sz_inds(sz) = sz_ind;
    end
    pt_data_clips.sz_inds(ipt) = {sz_inds};

end


% get soz localization (interested in temporal patients)
for i=1:(height(pt_data_clips))
    soz_loc = soz_info.region(contains(soz_info.name,num2str(pt_data_clips.ptID(i))));
    pt_data_clips.soz_loc(i) = soz_loc;
    pt_data_clips.temporal(i) = contains(lower(soz_loc),'temporal');
end 

%%

% set constants
plot_spec = 0;
fs = 6 ; % 6 sample every hour - units are in hours
prom_thresh = 0.2; % for grabbing peaks on periodogram
err = 0.1; % bounds around frequency of interest

% For inverse Wavelet Transform: Setting Constants from Torrence and Compo (TBL 2)
dt = 1/fs;
Cd = 0.776;
Psi0 = pi^(-.25);
dj = 1/12;
modifier = (dj*dt^(1/2))/(Cd*Psi0);
% Defining inverse wavelet function
invcwt = @ (wave,scale,pk_locs) modifier*sum(real(wave(pk_locs,:))./(scale(pk_locs)'.^(1/2)),1);


close all;

for ipt =1:length(ptIDs)

    ptID = ['HUP' num2str(ptIDs(ipt))];
    med_names = all_med_names{ipt};
    asm_load = all_pts_drug_samp{ipt};
    spikes = (all_spike_rate{ipt}+1);
    spikes = spikes./max(spikes);

    signal = zscore(spikes);
    exact_time = all_spike_times{ipt};
    time =linspace(0,max(exact_time),length(signal))./3600;


    % at least one medication tapered for >2days - pre/post condition breakpoints
    all_asm_inds = pt_data_clips.inds(ipt); all_asm_inds = all_asm_inds{:};
    dosing_periods = zeros(length(med_names),1);
    before_after_peak_periods = zeros(length(med_names),2);

    if ~all(all(isnan(all_asm_inds)))
        ind1 = max(all_asm_inds(:,2));
        ind2 = min(all_asm_inds(:,3));

        if (ind2-ind1) > taper_thresh
            exact_time = all_spike_times{ipt};
            time =linspace(0,max(exact_time),length(spikes))./3600;

            asm_before = asm_load(:,1:ind1);
            asm_after = asm_load(:,ind1+1:ind2);

            spikes_before = signal(:,1:ind1);
            spikes_after = signal(:,ind1+1:ind2);

            % determine dosing freqs in before period using wavelet transform
            for m =  1:length(med_names)
                sig = asm_before(m,:)';
                [pks,locs] = findpeaks(sig);
                target_dose_int = mean(diff(time(locs))); % in hours from med curve
                if ~(contains(med_names{m},'lorazepam')) && ~isempty(pks) && ~isnan(target_dose_int)
                    [wave,periods,~,~] = wt([time(1:ind1)',sig],hours(1/fs));
                    periodogram = mean(abs(wave).^2,2);
                    [pks,locs] = findpeaks(periodogram./max(periodogram)); %,'minpeakprominence',.3
                    if ~isempty(locs) && round(max(periods))>=20
                        [per_diff,loc_int] = min(abs(periods-target_dose_int));
                        assert(per_diff < target_dose_int*.33); % verify that it is a peak period in the signal - 33% error
                        dosing_freqs = (periods(loc_int));
                        dosing_periods(m) = round(dosing_freqs*10)./10;
                    end
                end
                % can plot this for figure to show that med is no longer dosed
                %                sig = asm_after(m,:)';
                %               [w,periods] = cwt(sig,(minutes(10)));
                %               [asm_after12_rec] = icwt(w,[],periods,range_12hr,SignalMean=mean(sig));
                %               [asm_after24_rec] = icwt(w,[],periods,range_24hr,SignalMean=mean(sig));

            end

            if plot_spec % plot the taper period, the asms, & spikes
                figure;clf; hold on %#ok<*UNRCH>
                subplot(2,1,1)
                plot(time,signal)
                xline(time(ind1),'r-','linewidth',3); xline(time(ind2),'r-','linewidth',3)
                xlabel('Time(hours)')
                subplot(2,1,2)
                plot(time, asm_load'); hold on;
                xline(time(ind1),'r-','linewidth',3); xline(time(ind2),'r-','linewidth',3)
                legend(med_names);title(ptID)
            end

            
            for p = 1:length(med_names) % for each med
                pk_per = dosing_periods(p);
                if pk_per > 0
                    [wave,period,scale,coi] = wt([time',signal'],hours(1/fs));

                    per_bounds = [pk_per-err*pk_per,pk_per+err*pk_per];
                    per_mask = period > per_bounds(1) & period < per_bounds(2);

                    % amplitude of periodogram peak pre/post - normalized
                    % across subjects
                    pre_periodogram = mean(abs(wave(:,1:ind1)).^2,2);
                    post_periodogram = mean(abs(wave(:,ind1:ind2)).^2,2);
                    pre_pk_amp = max(pre_periodogram(per_mask));
                    post_pk_amp = max(post_periodogram(per_mask));

                    before_after_peak_periods(p,:) = [pre_pk_amp post_pk_amp]; %paired samples
                end
            end
        end
    end
    pt_data_clips.dosing_periods(ipt) = {dosing_periods};
    pt_data_clips.before_after_dosing_periods(ipt) = {before_after_peak_periods};
end

%% test all spike amp in  pre/post dosing period:
close all; figure

before_afters =  pt_data_clips.before_after_dosing_periods(:);
all_before_after = vertcat(before_afters{:});
all_dosing_periods = pt_data_clips.dosing_periods(:); all_dosing_periods=vertcat(all_dosing_periods{:});
per_12hr_mask = abs(all_dosing_periods-12) < (12*.33); % meds that were dosed every 12hrs

med_label =  pt_data_clips.med_names(:);
med_label =  vertcat(med_label{:});
gaba_mask = contains(med_label,'lacos');
plot(all_before_after(gaba_mask,1),all_before_after(gaba_mask,2),'o'); hold on;
xlabel('before taper'); ylabel('after taper'); title('amplitude of spike rate at dosing frequency')
plot(0:40,0:40);
xlim([0 20]);ylim([0 20]); axis square;
% can we color code by medication?

% temporal patients only
temporal_pts = pt_data_clips(pt_data_clips.temporal==1,:);

before_afters =  temporal_pts.before_after_dosing_periods(:);
all_before_after = vertcat(before_afters{:});
all_dosing_periods = temporal_pts.dosing_periods(:); all_dosing_periods=vertcat(all_dosing_periods{:});
per_12hr_mask = abs(all_dosing_periods-12) < (12*.33); % meds that were dosed every 12hrs

med_label =  temporal_pts.med_names(:);
med_label =  vertcat(med_label{:});
gaba_mask = contains(med_label,'gabapentin');
figure;
plot(all_before_after(:,1),all_before_after(:,2),'o'); hold on;
xlabel('before taper'); ylabel('after taper'); title('amplitude of spike rate at dosing frequency: MTL patients')
plot(0:40,0:40);
xlim([0 20]);ylim([0 20]); axis square;

% try a linear mixed model to tease out the effect of medication taper on
% spikes at dosing frequency 




%% signal amplitude in dosing freq figure:
% % wave amplitude using phase in pre/post condition
% inverse_wavelet = invcwt(wave,scale,per_mask);
% invw_phase = angle(hilbert(inverse_wavelet));
%
% % condition masks
% pre_mask = time <= time(ind1); post_mask = time > time(ind1) & time < time(ind2);
%
% % breaking up phase into histograms
% [~,~,idxs] = histcounts(invw_phase,36);
% % Calculating x axis for plotting "folded" wavelets
% period_bins = splitapply(@(x) mean(x),invw_phase(pre_mask),findgroups(idxs(pre_mask)));
%
% % Calculating the average and std of signal in each phase bin within conditions
% pre_bin_means = splitapply(@(x) mean(x),inverse_wavelet(pre_mask),findgroups(idxs(pre_mask)));
% pre_bin_stds = splitapply(@(x) std(x),inverse_wavelet(pre_mask),findgroups(idxs(pre_mask)));
%
% post_bin_means = splitapply(@(x) mean(x),inverse_wavelet(post_mask),findgroups(idxs(post_mask)));
% post_bin_stds = splitapply(@(x) std(x),inverse_wavelet(post_mask),findgroups(idxs(post_mask)));
%
% % do a test at zero phase to see if the amplitude is affected
% [~,peak_bin]=(min(abs(period_bins)));
% post_groups = findgroups(idxs(post_mask));
% post_vals = inverse_wavelet(post_mask);
%
% pre_groups = findgroups(idxs(pre_mask));
% pre_vals = inverse_wavelet(pre_mask);
%
% %post condition vals at zero phase
% pre_zero_vals = pre_vals(pre_groups==peak_bin);
% post_zero_vals = post_vals(post_groups==peak_bin);
% [h,p,ci,stats]= ttest2(pre_zero_vals,post_zero_vals,'vartype','unequal')
%
%

%% PLOTTING CODE

% Plotting for signal amplitude in pre/post
% figure(); hold on;
% plot(period_bins, pre_bin_means,'LineWidth',1,'Color','blue','HandleVisibility','off');
% upper_bound = pre_bin_means + pre_bin_stds;
% lower_bound = pre_bin_means - pre_bin_stds;
% period_bins_flip = [period_bins,fliplr(period_bins)];
% fill_between = [upper_bound,fliplr(lower_bound)];
% fill(period_bins_flip,fill_between,'b','FaceAlpha',.5,'DisplayName',"Pre-Condition");
%
% plot(period_bins, post_bin_means, 'LineWidth',1,'Color','red','HandleVisibility','off');
% upper_bound = post_bin_means + post_bin_stds;
% lower_bound = post_bin_means - post_bin_stds;
% period_bins_flip = [period_bins,fliplr(period_bins)];
% fill_between = [upper_bound,fliplr(lower_bound)];
% fill(period_bins_flip,fill_between,'r','FaceAlpha',.5,"DisplayName","Post-Condition");
% xlabel('Phase (rad)')
% title(['Effect of ASM taper on Signal Amplitude: ' ptID])
% ylabel('Amplitude');
% legend("Location","best")
%
% % more optional plotting
%     figure;clf;
%     tiledlayout(3,1)
%     % Plot periodogram across whole timeseries
%     nexttile; hold on
%     % Calculate wavelet energy
%     all_periodogram = mean(abs(wave).^2,2);
%     plot(period,all_periodogram);
%     xlabel('Period (hrs)')
%     title('All Timeseries')
%     % Find the peaks (filter noise out with prominence)
%     [pks,locs] = findpeaks(all_periodogram,"MinPeakProminence",prom_thresh);
%     % if using different signal, will have to tune this prominence val
%     stem(period(locs),pks)
%     xticks(round(period(locs)*10)./10)
%     xlim([0,36])
%
%     % Plot periodogram in pre-condition
%     nexttile; hold on
%     pre_periodogram = mean(abs(wave(:,1:ind1)).^2,2);
%     plot(period,pre_periodogram);
%     xlabel('Period (hrs)')
%     title('Pre-Condition Periodogram')
%     [pks,pre_locs] = findpeaks(pre_periodogram,"MinPeakProminence",prom_thresh);
%     if ~isempty(pks)
%         stem(period(pre_locs),pks)
%     end
%     xticks(round(period(pre_locs)*10)./10)
%     xlim([0,36])
%
%     % Plot periodogram in post-condition
%     nexttile; hold on
%     post_periodogram = mean(abs(wave(:,ind1+1:ind2)).^2,2);
%     plot(period,post_periodogram);
%     xlabel('Period (hrs)')
%     title('Post-Condition Periodogram')
%     [pks,post_locs] = findpeaks(post_periodogram,"MinPeakProminence",prom_thresh);
%     stem(period(post_locs),pks)
%     xticks(round(period(post_locs)*10)./10)
%     xlim([0,36])
%
%     % plot the inverse wt of the freq of interest
%     figure;clf;
%     plot(time,signal,'LineWidth',.5); hold on
%     plot(time,inverse_wavelet*5,'LineWidth',1,'LineStyle','-')
%     legend("Raw Signal","12hr ICWT")
%
%
%
%     function waterplot(s,f,t)
%     % Waterfall plot of spectrogram
%     waterfall(f,t,abs(s)'.^2)
%     set(gca,XDir="reverse",View=[30 50])
%     xlabel("Frequency (Hz)")
%     ylabel("Time (s)")
%     end
%



