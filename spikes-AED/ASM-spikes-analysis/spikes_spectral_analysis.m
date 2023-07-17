%% spectral analysis of spike rate

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

plot_spec = 1;

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

        med_names = all_med_names{ipt};
        plot_dim = length(med_names)+1;
        
        % fft -> periodogram
        window =144;
        hrs =[1 2:48];
        f = (1./hrs) ./ 3600; % to units of samps/1sec (Hz)
        fs= 1/600;

        % before
        figure;
        plot_ind=1;
        for m = 1:length(med_names)
            subplot(plot_dim,2,plot_ind)
            plot_ind=plot_ind+1;
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
            
            subplot(plot_dim,2,plot_ind)
            plot_ind=plot_ind+1;
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


        subplot(plot_dim,2,plot_dim*2 -1)
        %x=spikes;
        x = spikes_before;
        [pxx,f]=periodogram(x,window*ones(1,length(x)),f,fs,'power');
        %convert f to cycle/hours and un-normalize it
        fplot =( f .* (3600 .* (hrs.^2))) ;
        plot(fplot,(pxx));
        xlabel('frequency (Hrs)')
        ylabel('dB')
        title([ ptID ' spikes (normalized); before taper'])

        subplot(plot_dim,2,plot_dim*2)
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
    end 
end

function waterplot(s,f,t)
% Waterfall plot of spectrogram
waterfall(f,t,abs(s)'.^2)
set(gca,XDir="reverse",View=[30 50])
xlabel("Frequency (Hz)")
ylabel("Time (s)")
end




