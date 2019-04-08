%%marks
time_tag = [273,1];
%%read steady state data
[std_num,std_txt,std_raw] = xlsread('Steady State.csv');
%%trim down the size of numerical data for sum 
trim_std_num = std_num(time_tag(1):end,time_tag(2):end);
trim_std_num(1,1) = 0;
%%calculation the normal value for sensors
dim_trim = size(trim_std_num);
for i = 2:dim_trim(2)
    avg = sum(trim_std_num) / dim_trim(1);
end
%%set threshold
l = 0.9 .*avg;
h = 1.1 .*avg;

%%read malfunction data
[mal_num,mal_txt,mal_raw] = xlsread('Malfunction 15 Fail Absorber Circulation Pump.csv');

dim_mal = size(mal_num);
%%initialize mandatory database
alarm = zeros(dim_mal(1),dim_mal(2));
alarm_log = cell(dim_mal(1),dim_mal(2));
log_cursor = [1,1];
%%detecting
for i = 2:dim_mal(2) %column
    for j = (time_tag(1) + 1):dim_mal(1) %row
        row_remap = j - time_tag(1) + 1;
        la = mal_num(j,i) < l(i);
        ha = mal_num(j,i) > h(i);
        if la == true||ha == true
            alarm(row_remap,i,la==true) = -1; %low
            alarm(row_remap,i,ha==true) = 1; %high
            alarm_log(log_cursor(1),log_cursor(2),log_cursor(1) == 1) = std_txt(time_tag(1),i);
            log_cursor(1) = log_cursor(1) + 1;
            alarm_log(log_cursor(1),log_cursor(2)) = num2cell(mal_num(j,1));
        end
    end
    log_cursor(1,2,log_cursor(1) ~= 1) = log_cursor(2) + 1; %increment
    log_cursor(1) = 1; %reset cursor
end