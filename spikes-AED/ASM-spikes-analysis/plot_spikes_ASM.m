close all;clear;

curr_path = pwd;
addpath([curr_path '/DATA'])
addpath([curr_path '/spikes-AED/aed_dose_modeling/figures_code'])
addpath([curr_path '/spikes-AED/aed_dose_modeling'])

tic


%load spike rate - new from 2/13/23 (samp/10min)
load('spikes_rates_021323.mat');
inds = cellfun('isempty',all_spike_rate);

cohort_info = readtable('HUP_implant_dates.xlsx');
ptIDs = cohort_info.ptID;
weights = cohort_info.weight_kg;
load('MAR_032122.mat')

% remove ptIDs that are nan
% inds = isnan(ptIDs);
% ptIDs(inds) =[];
% spike_file_dur(inds)=[];
% all_spike_rate(inds)=[];


[all_dose_curves,all_tHr,ptIDs,all_med_names,all_ieeg_offset,max_dur,emu_dur] = get_aed_curve_kg(ptIDs,weights);
%%
for ipt = 8%1:length(ptIDs)
    % patient medication administration
    figure;
    ptID =  ['HUP' num2str(ptIDs(ipt))];
    
    [med_names,meds,explant_date,starts_eeg,starts_emu] = parse_MAR(ptID,all_meds);
    
    offsets = all_ieeg_offset{2,ipt};
    ieeg_offset_datasets = all_ieeg_offset{1,ipt};
    
    % plot the spikes
    subplot(2,1,1)
    spike_rate=all_spike_rate{ipt}; %calculated spike rate for patient in list
    %spike_rate=log10(all_spike_rate{ipt}+1);
    time = all_spike_times{ipt};
    
    % align ieeg times for each file with emu medication times
    offset_vec = file_inds{ipt};
    for i = unique(offset_vec)'
        time_inds = (offset_vec==i);
        time(time_inds) = time(time_inds) - starts_eeg(i) + starts_emu(i);
    end 
    %time(offset_vec==1) = time(offset_vec==1) + offsets(1); %shift for t=0 to be start of emu stay, not start of ieeg recording
    time =time +offsets(1);
    plot(time./3600,spike_rate,'k','linewidth',.5); 
    %xlim([0 emu_dur(ipt)]);
    ylabel('spikes/10min'); xlabel('time (hrs)')
    title([ptID ': Spike rate and medication dose over EMU stay']);
    xlim([0 emu_dur(ipt)+24]);
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
    
    % plot dose curve
    subplot(2,1,2)
    %plot(0,0);
    h=zeros(1,length(all_med_names{ipt}));
    med_colors = lines(length(all_med_names{ipt}));
    for i=1:length(all_med_names{ipt})
        curve=all_dose_curves{ipt}{i};
        if ~isempty(curve)
            h(i)=plot(all_tHr{ipt}{i},(curve./max(curve)),'LineWidth',2,'Color',[med_colors(i,:) .5]);hold on;
            %h(i)=plot(all_Hr{ipt}{i},curve./max(curve));hold on; %normalized to [0 1]
        else
            all_med_names{ipt}(i)=[];
            h(i)=NaN;
        end
    end
    h(isnan(h))=[];
    % plot the seizures -- in ieeg times
    
    %get seizure times
    if ~isempty(offsets)
        for j =1:height(seizure_times)
            xline(seizure_times(j,1),'--r','linewidth',2);hold on;
        end
    end
    
    xlim([0 emu_dur(ipt)+24]);
    legend(all_med_names{ipt}');
    
end

%% save the plots in documents/Litt_lab
cd('/Users/ninaghosn/Documents/Litt_Lab/projects/Pioneer/AED-taper-networks/ASM_spikes/spike_asm_curves')
print_all_figures_to_eps()