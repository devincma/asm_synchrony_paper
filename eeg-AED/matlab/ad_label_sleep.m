function [sleep_labels,sleep_times,ptIDs] = ad_label_sleep(ptIDs)
% get and save spike rate. make into function?

sleep_labels = cell(1,length(ptIDs));
sleep_times = cell(1,length(ptIDs));

for x = 1:length(ptIDs)
    
    ptID = ptIDs{x};
    %load the spike data
    addpath('/Volumes/users/erinconr/projects/fc_toolbox/results/analysis/intermediate/');
    fname = [ptID '.mat'];
    try
        load(fname);
        % get ad and labels from the structure
        ad = summ.ad;
        labels = summ.labels;
        
        % Get non intracranial labels
        ekg = find_non_intracranial(labels);
        
        % remove non intracranial labels
        ad = ad(~ekg,:);
        
        % average across electrodes to get single value for each time
        ad = nanmean(ad,1);
        
        % normalize across times
        norm_ad = (ad-nanmedian(ad))./iqr(ad);
        
        % get the sleep/wake cut-off (seemed to do the best job in a test of 10 patients)
        disc = -0.4054;
        
        % classify as sleep or wake
        sleep_labels{x} = norm_ad > disc; % 1 = wake, 0=sleep
        sleep_times{x}=linspace(1,length(norm_ad)./6,length(norm_ad));

        % sleep = norm_ad <= disc;
        % wake = norm_ad > disc;
        
    catch
        ptIDs{x}=[];
    end
end
