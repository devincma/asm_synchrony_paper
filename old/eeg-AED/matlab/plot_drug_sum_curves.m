close all;clear;
addpath('/Volumes/USERS/nghosn3/Pioneer/DATA');
addpath('/Volumes/USERS/nghosn3/Pioneer/spikes-AED/aed_dose_modeling')
addpath('/Volumes/USERS/nghosn3/Pioneer/spikes-AED/aed_dose_modeling/figures_code')
cd('/Volumes/USERS/nghosn3/Pioneer/spikes-AED');


% loadthe aed metadata
aed_params = readtable('AED_metadata.xlsx');
aed_params.medication = arrayfun(@lower,aed_params.medication);

% Get which patients have AED data, load the data
cohort_info = readtable('HUP_implant_dates.xlsx');
ptIDs = cohort_info.ptID;
weights = cohort_info.weight_kg;

ptID = 138;
weight = weights(ptIDs==ptID);
% get their medication data
[all_dose_curves,all_tHr,ptIDs,all_med_names,all_ieeg_offset,max_dur,emu_dur] = get_aed_curve_kg(ptID,weight);

% load home medications
load('MAR_032122.mat')
%%

for ipt = 1:length(ptIDs)
    % patient medication administration
    ptID = ['HUP' num2str(ptIDs(ipt))];
    offsets = all_ieeg_offset{2,ipt};
    ieeg_offset_datasets = all_ieeg_offset{1,ipt};
    
    med_names = all_med_names{ipt};
    tHr = all_tHr{ipt};
    pt_curves = all_dose_curves{ipt};
    
    figure('Position', [10 10 900 300])
    drugs =zeros(length(med_names),ceil(emu_dur(ipt)*60));
    for n =1:length(med_names)
        drug=pt_curves{n};
        drug=drug./nanmax(drug); %normalize each drug curve
        if ~isempty(drug)
            dStart = round(tHr{n}(1)*60)-1;
            drugs(n,dStart+1:dStart+length(drug))=drug;
        end
    end
    drug_sum=nansum(drugs,1);
    drug_sum=drug_sum./length(med_names);
    drug_sum(drug_sum==0)=[];
    
    tstart = min(cellfun(@(x) x(1),tHr));
    tend =  min(cellfun(@(x) x(end),tHr));
    time = linspace(tstart,tend,length(drug_sum));
    plot(time,drug_sum); hold on;
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
    
    ylabel('AED BPL (normalized)');xlabel('time (Hr)')
    ylim([0 1]);
   
       
    title(ptID);
    save_path='/Users/ninaghosn/Documents/Litt_Lab/projects/Pioneer/AED-taper-networks/results-figures/';
     print([save_path 'drug_sum_HUP',num2str(ptIDs(ipt)),'.eps'],'-depsc2','-painters', '-tiff', '-r300', '-f')
    
    
end
