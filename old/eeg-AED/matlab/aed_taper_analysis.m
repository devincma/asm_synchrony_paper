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

% get the seizure and SOZ localization information 
soz_info = readtable('Erin_szs_times.xlsx','VariableNamingRule','preserve','Sheet','SOZ');

% load home medications and medication data
load('home_meds.mat');
load('MAR_032122.mat')

% get AED curves and taper info 
[all_dose_curves,all_Hr,ptIDs,all_med_names,ieeg_offset,~,emu_dur] = get_aed_curve_kg(ptIDs,weights);

%% 
all_rates = cell(1,length(ptIDs));
all_taper_periods =  cell(1,length(ptIDs));
all_taper_meds =  cell(1,length(ptIDs));
all_start_end_doses =  cell(1,length(ptIDs));
all_day1_change =  cell(1,length(ptIDs));


for i=1:length(ptIDs)
    ptID = ['HUP' num2str(ptIDs(i))];
    pt_curves = all_dose_curves{i};
    tHr = all_Hr{i};
    implant_date = cohort_info.Implant_Date(cohort_info.ptID == ptIDs(i));
    disp("here");
    disp(ptID);
    [med_names,meds,explant_date] = parse_MAR(ptID,all_meds);
    % [taper_info,any_taper] = get_taper_info(meds, med_names);
    [rates,taper_period,tapered_med_names,start_end_doses,day1_change] = get_taper_speed(pt_curves,tHr,meds,med_names,implant_date);
    
    all_rates{i} = rates;
    all_taper_periods{i} = taper_period ;
    all_taper_meds{i} = tapered_med_names ;
    all_start_end_doses{i} =  start_end_doses;
    all_day1_change{i} = day1_change;

end

%% get average decrease in all meds in the first 24 hrs

% for the patients that have severity scores, get their overall taper
% agressiveness score 

% Get the patients that also have seizure severity scores
load('seizure_meta_data_with_ativan_times.mat')
sz_pts = unique(all_sz_data.Patient);
pt_inds=zeros(1,length(ptIDs));
for x = 1:length(ptIDs)
    ptID = ['HUP' num2str(ptIDs(x))];
    pt_inds(x) = sum(contains(sz_pts,ptID));
end

severity_ptIDs = ptIDs(logical(pt_inds));
severity_rates = all_rates(logical(pt_inds));
mean_severity_rates = cellfun(@nanmedian,severity_rates,'UniformOutput',false);
mean_severity_rates = vertcat(mean_severity_rates{:});

taper_coeff = mean_severity_rates(:,2);
taper_rate =   mean_severity_rates(:,1);


pt_severity_scores = nan(1,length(severity_ptIDs));
for i =1:length(severity_ptIDs)
    inds = contains(all_sz_data.Patient,num2str(severity_ptIDs(i)));
    pt_severity_scores(i)=nanmean(all_sz_data.seizure_severity_distance(inds)); %add all severity scores for an overall seizure severity 'burden'- takes into account #sz
end 

% remove nans
nan_inds = isnan(taper_coeff);
taper_coeff(nan_inds)=[];
taper_rate(nan_inds)=[];
pt_severity_scores(nan_inds)=[];
length_stay = emu_dur(logical(pt_inds));
length_stay(nan_inds)=[];

x = pt_severity_scores'; y = (taper_coeff);
[r,p] = corr([x y]); %coefficient for exp fit
r = r(2,1);
p = p(2,1);

figure;
subplot(2,2,1)
scatter(x,y); hold on; axis square; title('severity score vs taper speed (exp coeff)')
coefs = polyfit(x,y,1);
aed_pred = (coefs(1)*x) +coefs(2);
plot(x,aed_pred,'k','Linewidth',2);
legend(['p: ' num2str(p) ', R: ' num2str(r)]);

x = pt_severity_scores'; y = taper_rate;
[r,p] = corr([x y]); %coefficient for exp fit
r = r(2,1);
p = p(2,1);

subplot(2,2,2)
scatter(x,y); hold on; axis square; title('severity score vs taper speed (exp rate)')
coefs = polyfit(x,y,1);
aed_pred = (coefs(1)*x) +coefs(2);
plot(x,aed_pred,'k','Linewidth',2);
legend(['p: ' num2str(p) ', R: ' num2str(r)]);

% duration of EMU stay and seizure severity 
x = length_stay'; y = taper_coeff;
[r,p] = corr([x y]); %coefficient for exp fit
r = r(2,1);
p = p(2,1);

subplot(2,2,3)
scatter(x,y); hold on; axis square; title('length of stay vs taper speed (exp coef)')
coefs = polyfit(x,y,1);
aed_pred = (coefs(1)*x) +coefs(2);
plot(x,aed_pred,'k','Linewidth',2);
legend(['p: ' num2str(p) ', R: ' num2str(r)]);

x = length_stay'; y = taper_rate;
[r,p] = corr([x y]); %coefficient for exp fit
r = r(2,1);
p = p(2,1);

subplot(2,2,4)
scatter(x,y); hold on; axis square; title('length of stay vs taper speed (exp rate)')
coefs = polyfit(x,y,1);
aed_pred = (coefs(1)*x) +coefs(2);
plot(x,aed_pred,'k','Linewidth',2);
legend(['p: ' num2str(p) ', R: ' num2str(r)]);

% length of stay vs severity 
x = pt_severity_scores'; y = length_stay';
[r,p] = corr([x y]); %coefficient for exp fit
r = r(2,1);
p = p(2,1);

figure;
scatter(x,y); hold on; axis square; title('severity score vs length of emu stay')
coefs = polyfit(x,y,1);
aed_pred = (coefs(1)*x) +coefs(2);
plot(x,aed_pred,'k','Linewidth',2);
legend(['p: ' num2str(p) ', R: ' num2str(r)]);

%print(['taper_severity.eps'],'-depsc2','-painters', '-tiff', '-r300', '-f')

%% 
[tapered_pts,all_taper_info] = get_tapered_patients(ptIDs,all_meds);


