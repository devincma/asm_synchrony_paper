close all;clear;

addpath('/Volumes/USERS/nghosn3/Pioneer/spikes-AED')
addpath('/Volumes/USERS/nghosn3/Pioneer/DATA')
load('patient_localization_final.mat');
load('synchronizability_HUP.mat')

fname = 'DATA_MASTER.json'; 
fid = fopen(fname); raw = fread(fid,inf); str = char(raw'); 
fclose(fid); 
data_master = jsondecode(str);


ptIDs = {'HUP173','HUP157','HUP160','HUP181','HUP190','HUP144','HUP146','HUP140','HUP164','HUP177','HUP187','HUP166'};

lens = zeros(1,length(ptIDs));
%loop through patients 
for y = 2:length(ptIDs)
    
   
    %find patient in struct
    for k = 1:length(patient_localization)
        if patient_localization(k).patient == ptIDs{y}
            pt_ind = k; end
    end
    ptID = ptIDs{y};
    
    % load some data
    cd('/Volumes/USERS/pattnaik/hmm-emu-state-space/data')
    addpath([cd '/' ptID])
    load('bandpower-windows-5win.mat'); %contains the time vector needed for plotting
    tSec = entireT*1e-6;
    %get their seizure times 
    seizures = data_master.PATIENTS.(ptID).Events.Ictal;
    seizure_times =[];
    vars = fields(seizures);
    for j =1:length(vars)
        start = seizures.(vars{j}).SeizureEEC;
        try stop = seizures.(vars{j}).SeizureEnd;
        catch; stop =NaN;
        end
        seizure_times = [seizure_times;start stop];
    end 
    seizure_times = seizure_times ./3600; %convert to hours
    cd('/Volumes/USERS/pattnaik/hmm-emu-state-space/data') 
    addpath([cd '/' ptID])
    %load('connectivity-windows-5win'); %load the connectivity matrices
    load('target-electrodes-regions.mat'); %load the target electrode names used in PC
    %some properties, and soz/nonsoz channels
    elecs = zeros(1,length(patient_localization(pt_ind).soz));
    elecs(targetElectrodesRegionInds) =1; %for getting labels of used electrodes
    soz_elecs = patient_localization(pt_ind).soz;
    soz_elecs_target = logical(soz_elecs(targetElectrodesRegionInds));
    nChannels =  height(Regions);
    labels = patient_localization(pt_ind).labels; elecLabels = labels(targetElectrodesRegionInds);
    
    
    addpath([cd '/DATA/med-data'])
    meds = readtable([ptID '_meds.xlsx']);
    days = unique(meds.day);
    med_names = unique(meds.medication);
    med_times = [meds.start]./3600; %start times in seconds, convert to hr for plotting
    nan_inds = isnan(med_times); med_times(nan_inds) = 0;
    
    seizure_times_plot = seizure_times+ days(1)*24; %shift for plotting to align with EMU DAY#
    
    
    %get total dose per day, including acute drugs for breakthrough seizures
    each_drug_dose = zeros(length(days),length(med_names));
    for i=1:length(days)
        day_inds = find(meds.day == days(i));
        for x=1:length(med_names)
            med_inds = find(contains(meds.medication(day_inds),med_names(x)));
            med_doses = meds.dose(day_inds);
            doses = cell2mat(med_doses(med_inds)');
            if ~isempty(doses)
                doses=replace(doses,'mg',',');doses=strsplit(doses,',');
                doses = cell2mat(cellfun(@str2num,doses,'uniform',0));
                each_drug_dose(i,x)=sum(doses);
            else
                each_drug_dose(i,x)=0;
            end
        end
        
    end
    %%
    total_dose = sum(each_drug_dose,2); %including ativan
    
    %plot doses of medication over EMU stay- some include alcohol
    %administered
    figure();
    subplot(2,1,2)
    days_norm = linspace(0,1,length(days));
    plot(days,each_drug_dose,'.-'); hold on;
    plot(days,total_dose,'k','LineWidth',1.5);
    legend([med_names; {'total AED dosage'}])
      %plot seizure times
    for j =1:length(seizure_times_plot)
        xline(seizure_times_plot(j,1)./24,'r');hold on;
    end
    hold off;
    
    %plot the synchronizability 
    temp_sync = all_S{y,1};
    lens(y) = length(temp_sync);
    tVec = linspace(0,1,lens(y));
    subplot(2,1,1)
    plot(tSec./(3600*24),smoothdata(temp_sync,'gaussian')+(y)); hold on;
    ylabel('synchronizability'); xlabel('time (days)')
    legend();title('synchronizability over time');
    %plot seizure times
    for j =1:length(seizure_times_plot)
        xline(seizure_times_plot(j,1)./24,'r');hold on;
    end
    
    title([ptID ': AED mg/day during EMU stay']); xlabel('day# of EMU stay');
    legend([{'total AED dosage'},{'syncronizability'},{'Seizures'}])
    
    
end 

