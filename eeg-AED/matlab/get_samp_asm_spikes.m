function [all_pts_drug_samp, all_pts_tvec] = get_samp_asm_spikes(meds_fname,spikes_fname,all_ieeg_offset, all_dose_curves, all_tHr,ptIDs)

load(spikes_fname,'all_spike_rate','all_spike_times','file_inds');
load(meds_fname,'all_meds');

all_pts_drug_samp = cell(1,length(ptIDs));
all_pts_tvec = cell(1,length(ptIDs));

for ipt = 1:length(ptIDs)

    ptID =  ['HUP' num2str(ptIDs(ipt))];
    [med_names,~,~,starts_eeg,starts_emu] = parse_MAR(ptID,all_meds);

    offsets = all_ieeg_offset{2,ipt};
    spike_rate = all_spike_rate{ipt}; %calculated spike rate for patient in list
    %spike_rate=log10(all_spike_rate{ipt}+1);
    time = all_spike_times{ipt};

    % align ieeg times for each file with emu medication times
    offset_vec = file_inds{ipt};
    for i = unique(offset_vec)'
        time_inds = (offset_vec==i);
        time(time_inds) = time(time_inds) - starts_eeg(i) + starts_emu(i);
    end
    time = (time + offsets(1))/3600; %shift for t=0 to be start of emu stay, not start of ieeg recording. convert to hours

    %sample the med curves to be time aligned with the spikes
    pt_drug_curves = all_dose_curves{ipt};
    pt_tHr = all_tHr{ipt};
    drugs_samp =zeros(length(med_names),length(spike_rate)); %450 hours of EMU stay in minutes
    for i =1:length(med_names)
        drug=pt_drug_curves{i};
        drug=drug./nanmax(drug); %normalize each drug curve
        drug_samp = zeros(1,length(spike_rate));
        if ~isempty(drug)
            for t = 1:length(time)
                if time(t)-pt_tHr{i}(1) >0
                    [~,t_ind] = min(abs(time(t)-pt_tHr{i})); % find closest time point
                    drug_samp(t) = drug(t_ind);
                end
            end
            drugs_samp(i,:)=drug_samp;
        end
    end
    assert(length(time) == length(drugs_samp));
    all_pts_drug_samp(ipt) = {drugs_samp};
    all_pts_tvec{ipt} = {time};


end



end