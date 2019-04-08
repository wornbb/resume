cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\Feature Extraction\006');
load('filter_bank.mat');
sei = categorical({'seizure'});
non = categorical({'nonseizure'});

min_f = 1;
max_f = 24;
interval_f = 3;
tappers = round((max_f-min_f)/interval_f);

cd D:\DataBase\NonSeizure
load('NonSeizure_chb01_03_1');
cd D:\DataBase\Seizure
load('HaveSeizure_chb01_03_1');
% selected_channel = [1 2 3 4 5 6 8 9]; legacy when testing
seizure = clean(seizure(:,:),'offline');
table_sei = real_time_simu_2s(seizure,@fx006,sei,filter_bank);

non_seizure = clean(non_seizure(:,:),'offline');
table_non = real_time_simu_2s(non_seizure,@fx006,non,filter_bank);

Tfft = [table_sei;table_non];
cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\Model Tranning\006');
save('Tfft2');

cd D:\DataBase\Seizure
seizure_name = ls;
for j = 19:180
    cd D:\DataBase\Seizure
    load(seizure_name(j,:)); 
    seizure = clean(seizure(:,:),'offline');
    table_sei = real_time_simu_2s(seizure,@fx006,sei,filter_bank);
    Tfft = [Tfft;table_sei];
    cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\Model Tranning\006');
save('Tfft2');
end
cd D:\DataBase\NonSeizure
non_sei_name = ls;
for j = 4:18
    cd D:\DataBase\NonSeizure
    load(non_sei_name(j,:)); 
    [l,~] = size(non_seizure);
    if l>=121600
        non_seizure = clean(non_seizure,'offline');
    else
        non_seizure = clean(non_seizure,'offline');
    end
    table_non = real_time_simu_2s(non_seizure,@fx006,non,filter_bank);
    Tfft = [Tfft;table_non];
    cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\Model Tranning\006');
save('Tfft2');
end %j = 8
cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\Model Tranning\006');
save('Tfft2');

