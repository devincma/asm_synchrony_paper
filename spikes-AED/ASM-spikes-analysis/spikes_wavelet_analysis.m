%% spectral analysis of spike rate using wavelet transform

close all;clear;
nina_path = '/Volumes/USERS/nghosn3/Pioneer';
addpath([nina_path '/DATA'])
addpath([nina_path '/spikes-AED/aed_dose_modeling/figures_code'])
addpath([nina_path '/spikes-AED/aed_dose_modeling'])

devin_path = '/Volumes/USERS/devinma/Pioneer';
addpath([devin_path '/spikes-AED/ASM-spikes-analysis'])

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

%% find patients for spectral analysis: 1) taper and 'off' of at least one to all medications for >2days
pt_data_clips = table();
pt_data_clips.ptID = (ptIDs);
pt_data_clips.med_names = all_med_names';
pt_data_clips.inds = cell(length(ptIDs),1);

days=2; % threshold in days
taper_thresh = days*(24*60)./10; % 10min segments in 2 day period

for ipt =1:length(ptIDs)
    asm_curves = all_pts_drug_samp{ipt};
    med_names = all_med_names{ipt};

    %find when each drug was stopped and find period of halt
    inds = nan(length(med_names),3);
    for n=1:length(med_names)
        med_curve = asm_curves(n,:);
        [pks,locs] = findpeaks(med_curve);
        diffs = diff(locs);
        last_admin = find(diffs >= taper_thresh);
        if ~isempty(last_admin)
            stop_med =locs(last_admin(1)); %ind+1 of last administration
            restart_med =locs(last_admin(1)+1)-1; % ind -1 of next peak after stopping

            inds(n,1) = 1;
            inds(n,2) = stop_med;
            inds(n,3) = restart_med;
        end
    end
    pt_data_clips.inds(ipt) = {inds};
end


%%

plot_spec = 0;

for ipt =75%1:length(ptIDs)

    ptID = ['HUP' num2str(ptIDs(ipt))];
    asm_load = all_pts_drug_samp{ipt};
    spikes = all_spike_rate{ipt}; spikes = spikes./max(spikes);

    % split the asm load and spikes to before and after taper (no more
    % drug) - manually selected
    all_asm_inds = pt_data_clips.inds(ipt); all_asm_inds = all_asm_inds{:};

    % If the patient had at least one medication tapered for >2days
    if ~all(all(isnan(all_asm_inds)))
        ind1 = max(all_asm_inds(:,2));
        ind2 = min(all_asm_inds(:,3));

        if (ind2-ind1) > taper_thresh
            asm_before = asm_load(:,1:ind1);
            asm_after = asm_load(:,ind1+1:ind2);

            spikes_before = spikes(:,1:ind1);
            spikes_after = spikes(:,ind1+1:ind2);
            exact_time = all_spike_times{ipt};
            time =linspace(0,max(exact_time),length(signal))./3600; 

            med_names = all_med_names{ipt};
            plot_dim = length(med_names)+1;

            % try wavelet transform and extract frequency of dosing -

            % determine dosing frequency from wavelet scalogram
            range_6hr = minutes([4 8]).* 60;
            range_12hr = minutes([10 14]).* 60;
            range_24hr = minutes([20 26]).* 60;
            figure;
            plot_ind=1;
            figure;
            for m = 1:length(med_names)

                sig = asm_before(m,:)';
                [w,periods] = cwt(sig,(minutes(10)));
                [asm_before12_rec] = icwt(w,[],periods,range_12hr,SignalMean=mean(sig));
                %[asm_before24_rec] = icwt(w,[],periods,range_24hr,SignalMean=mean(sig));

                if plot_spec
                    subplot(plot_dim,2,plot_ind)
                    plot_ind = plot_ind +1;
                    plot(asm_before12_rec)
                    %plot(asm_before24_rec)
                    hold on;plot(sig)
                    title(med_names{m})
                    legend('reconstruction 12hr','og')
                end


                sig = asm_after(m,:)';
                [w,periods] = cwt(sig,(minutes(10)));
                [asm_after12_rec] = icwt(w,[],periods,range_12hr,SignalMean=mean(sig));
                %[asm_after24_rec] = icwt(w,[],periods,range_24hr,SignalMean=mean(sig));

                if plot_spec
                    subplot(plot_dim,2,plot_ind)
                    plot_ind = plot_ind +1;
                    plot(asm_after12_rec)
                    %plot(asm_after24_rec)
                    hold on;plot(sig)
                    title(med_names{m})
                    legend('reconstruction 12hr','og')

                end
            end



            % inverse wavelet transform of spikes
            sig = spikes_before;
            [w,periods] = cwt(sig,(minutes(10)));
            [spikes_before6_rec] = icwt(w,[],periods,range_6hr,SignalMean=mean(sig));
            [spikes_before12_rec] = icwt(w,[],periods,range_12hr,SignalMean=mean(sig));
            [spikes_before24_rec] = icwt(w,[],periods,range_24hr,SignalMean=mean(sig));

    
            sig = spikes_after;
            [w,periods] = cwt(sig,(minutes(10)));
            [spikes_after6_rec] = icwt(w,[],periods,range_6hr,SignalMean=mean(sig));
            [spikes_after12_rec] = icwt(w,[],periods,range_12hr,SignalMean=mean(sig));
            [spikes_after24_rec] = icwt(w,[],periods,range_24hr,SignalMean=mean(sig));

            pk_per = 12;

            % Plotting inverse wavelet and signal (scales changed for comparison)
            figure;clf;
            tiledlayout(2,1); nexttile; hold on
            plot(time(1:ind2),spikes(1:ind2),'LineWidth',.5)
            plot(time(1:ind2),[spikes_before12_rec,spikes_after12_rec] ,'LineWidth',2,'LineStyle','-')
            legend({'Raw Signal','12Hr ICWT'})
            nexttile; hold on
            plot(time(1:ind2),spikes(1:ind2),'LineWidth',.5)
            plot(time(1:ind2),[spikes_before12_rec,spikes_after12_rec],'LineWidth',2,'LineStyle','-')
            xlim([0,50])

            if plot_spec
                subplot(plot_dim,2,plot_dim*2 -1)
                plot(spikes_before6_rec,'linewidth',3); hold on;
                plot(spikes_before12_rec,'linewidth',3);hold on;
                plot(spikes_before24_rec,'linewidth',3)
                hold on;plot(sig)
                title([ ptID ' spikes (normalized); before taper'])
                legend('reconstruction 6hr','reconstruction 12hr','reconstruction 24hr','og')

                subplot(plot_dim,2,plot_dim*2)
                plot(spikes_after6_rec,'linewidth',3);hold on;
                plot(spikes_after12_rec,'linewidth',3)
                plot(spikes_after24_rec,'linewidth',3)
                hold on;plot(sig)

                title([ ptID ' spikes (normalized); after taper (no meds)'])
                hold on;
            end
        end
    end
end


function waterplot(s,f,t)
% Waterfall plot of spectrogram
waterfall(f,t,abs(s)'.^2)
set(gca,XDir="reverse",View=[30 50])
xlabel("Frequency (Hz)")
ylabel("Time (s)")
end




