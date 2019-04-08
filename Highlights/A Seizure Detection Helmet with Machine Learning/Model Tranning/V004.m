
sei = categorical({'seizure'});
non = categorical({'nonseizure'});
% min_f = 3;
% max_f = 15;
% interval_f = 0.68;
min_f = 1;
max_f = 24;
interval_f = 1;
tappers = round((max_f-min_f)/interval_f);

cd D:\DataBase\NonSeizure
load('NonSeizure_chb01_01');
cd D:\DataBase\Seizure
load('HaveSeizure_chb01_04_1');
selected_channel = [1 2 3 4 5 6 8 9];
seizure = clean(seizure(:,selected_channel),'offline');
table_sei = real_time_simu(seizure,@fx004,sei);
% [l,~] = size(seizure);
% t = fix(l/256);
% buffer = zeros(256*2,24);
% buffer = array2table(buffer);
% selected_channel = 1:8;
% for i  = 1:2:(t-1)
%     data = seizure(1+256*(i-1):256*(i+1),selected_channel);
%     buffer{:,1:16} = buffer{:,9:24};
%     buffer{:,17:24} = data{:,:};
%     table_sei = fx003(buffer,sei,min_f,max_f);
% end
non_seizure = clean(non_seizure(:,selected_channel),'offline');
table_non = real_time_simu(non_seizure,@fx004,non);

Tfft = [table_sei;table_non];

% figure;
% hold on;
% [l c] = size(Tfft);
% interval = l/4;
% scatter(Tfft{1:interval,1},Tfft{interval+1:interval*2,1},'r');
% scatter(Tfft{1 + interval*2:interval*3,1},Tfft{interval*3+1:interval*4,1},'b');

seizure_name = ls;
for j = 3:18
    load(seizure_name(j,:)); 
    seizure = clean(seizure(:,selected_channel),'offline');
    table_sei = real_time_simu(seizure,@fx004,sei);
    Tfft = [Tfft;table_sei];
end
cd D:\DataBase\NonSeizure
non_sei_name = ls;
for j = 3:18
    load(non_sei_name(j,:)); 
    non_seizure = clean(non_seizure(:,selected_channel),'offline');
    table_non = real_time_simu(non_seizure,@fx004,non);
    Tfft = [Tfft;table_non];
end
cd('C:\Users\shen1\Desktop\develoopment\Model Tranning\temp');
save('Tfft');

