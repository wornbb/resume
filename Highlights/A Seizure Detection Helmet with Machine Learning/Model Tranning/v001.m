
sei = categorical({'seizure'});
non = categorical({'nonseizure'});


cd D:\DataBase\NonSeizure
load('NonSeizure_chb01_01');
cd D:\DataBase\Seizure
load('HaveSeizure_chb01_04_1');

seizure.type(:,1) = sei;
non_seizure.type(:,1) = non;
TrainData = [seizure;non_seizure];

seizure_name = ls;
for i = 3:18
   load(seizure_name(i,:)); 
    seizure.type(:,1) = sei;
    TrainData = [TrainData;seizure];
end

TrainTable.type = TrainData.type;