close all;clear;

% addpath('/mnt/leif/littlab/users/nghosn3/Pioneer/spikes-AED/aed_dose_modeling/figures_code')
% addpath('/mnt/leif/littlab/users/nghosn3/Pioneer/DATA');
% addpath('/mnt/leif/littlab/users/nghosn3/Pioneer/spikes-AED/aed_dose_modeling')
 addpath('/Volumes/USERS/nghosn3/Pioneer/spikes-AED/aed_dose_modeling/figures_code')
 addpath('/Volumes/USERS/nghosn3/Pioneer/DATA');
 addpath('/Volumes/USERS/nghosn3/Pioneer/spikes-AED/aed_dose_modeling')


%load spike rate - new from 10/22/21 (samp/10min) - cheeck what datasets are contained
load('all_spike_rate_new.mat');
tic
% Get which patients
cohort_info = readtable('HUP_implant_dates.xlsx');
ptIDs = cohort_info.ptID;
weights = cohort_info.weight_kg;

% plot one patient
ptID = 224;
ptIDs = ptIDs(ptIDs ==ptID);
weights = weights(ptIDs ==ptID); % cross checking with nlp data, using 70kg as mean


[all_dose_curves,all_Hr,ptIDs,all_med_names,ieeg_offset] = get_aed_curve_kg(ptIDs,weights); % 

%%
% get subplot dims
% n = 2;
% m = ceil(length(ptIDs)./n);

for ipt = 1:length(ptIDs)
    % patient medication administration
    ptID = ['HUP' num2str(ptIDs(ipt))];
     offsets = ieeg_offset{2,ipt};
     ieeg_offset_datasets = ieeg_offset{1,ipt};
%     
    % plot dose curve
    % subplot(m,n,ipt)
    figure('Position', [10 10 900 300])
    %plot(0,0);
    h=zeros(1,length(all_med_names{ipt}));
    med_colors = lines(length(all_med_names{ipt}));
    for i=1:length(all_med_names{ipt})
        curve=all_dose_curves{ipt}{i};
        if ~isempty(curve)
            h(i)=plot(all_Hr{ipt}{i},curve./max(curve),'LineWidth',2,'Color',[med_colors(i,:) .5]);hold on;
            %h(i)=plot(all_Hr{ipt}{i},curve./max(curve));hold on; %normalized to [0 1]
        else
            all_med_names{ipt}(i)=[];
            h(i)=NaN;
        end
    end
    h(isnan(h))=[];
%    plot the seizures -- in ieeg times
    
    %get seizure times
    [seizure_times,seizure_dataset] = get_seizure_times_from_sheet(ptID);
    if ~isempty(offsets)
        for j =1:height(seizure_times)
            % check which dataset the seizure is from, and add appropriate offset
            if isequal(seizure_dataset{j},'D01') || isequal(seizure_dataset{j},'one file')
                seizure_times(j,1)= (offsets(1)+(seizure_times(j,1)))./3600;
            else
                %ind = str2double(seizure_dataset{j}(end));
                ind = contains(ieeg_offset_datasets,['D0' seizure_dataset{j}(end)]);
                dataset_offset = offsets(ind);
                seizure_times(j,1)= (seizure_times(j,1) + dataset_offset)./3600; %convert to hours
            end
            xline(seizure_times(j,1),'--r','linewidth',2);hold on;
        end
    end
    % convert seizure times to indices, so to minutes
    seizure_inds = round(seizure_times(:,1) *60);
%     
    ylabel('AED BPL (normalized)');xlabel('time (Hr)')
    %xlim([0 time(end)]);
    if ~isempty(seizure_times)
        legend(h(:),all_med_names{ipt}');
    else
        legend(all_med_names{ipt}');
    end
    legend(all_med_names{ipt}');
    title(ptID);
    
     save_path='/Users/ninaghosn/Documents/Litt_Lab/projects/Pioneer/AED-taper-networks/updates_progress/4-22/bpl_curves_4-29/';
     %print([save_path 'all_drugs_HUP',num2str(ptIDs(ipt)),'.eps'],'-depsc2','-painters', '-tiff', '-r300', '-f')
end
toc

