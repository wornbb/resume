
sei = categorical({'seizure'});
non = categorical({'nonseizure'});


cd D:\DataBase\NonSeizure
load('NonSeizure_chb01_01');
cd D:\DataBase\Seizure
load('HaveSeizure_chb01_04_1');

seizure.type(:,1) = sei;
fft_sei = varfun(@fft,seizure(:,1:22));
amp_sei = varfun(@abs,fft_sei);
table_sei = amp_sei;
table_sei.type = seizure.type;

non_seizure.type(:,1) = non;
fft_non = varfun(@fft,non_seizure(:,1:22));
amp_non = varfun(@abs,fft_non);
table_non = amp_non;
table_non.type = non_seizure.type;
Tfft = [table_sei;table_non];


seizure_name = ls;
for j = 3:18
   load(seizure_name(j,:)); 
    seizure.type(:,1) = sei;
    fft_sei = varfun(@fft,seizure(:,1:22));
    amp_sei = varfun(@abs,fft_sei);
    table_sei = amp_sei;
    table_sei.type = seizure.type;
    Tfft = [Tfft;table_sei];
end



