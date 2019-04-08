
function y = AlarmSystemV23(number,mal)
    close all
    %%marks
    time_tag = [273,1];
    %%read steady state data
    [std_num,std_txt,std_raw] = xlsread('Steady State.csv');
    std_txt = string(std_txt);
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
    %%filename = uigetfile('*.csv');
    filename = sprintf('Malfunction %d.csv',mal);
    [mal_num,mal_txt,mal_raw] = xlsread(filename);  
    trim_mal_num = mal_num(time_tag(1):end,time_tag(2):end);
    dim_mal = size(trim_mal_num);
    %%initialize mandatory database
    alarm = struct('title',string([]),'index',zeros(1,dim_mal(2)),'value',zeros(dim_mal(1),dim_mal(2)),'type',zeros(dim_mal(1),1));
    log_cursor = [1,1];
    %%detecting alarm series
    for i = 2:dim_mal(2) %column
        for j = 1:dim_mal(1) %row
            la = trim_mal_num(j,i) < l(1,i);
            ha = trim_mal_num(j,i) > h(1,i);
            if la == true||ha == true
                log_cursor(1) = log_cursor(1) - 1;
                alarm.title(1,log_cursor(2)) = std_txt(time_tag(1),i);
                alarm.index(1,log_cursor(2)) = i;
                alarm.value(:,log_cursor(2)) = trim_mal_num(:,i);
                break;
            end
        end
        if log_cursor(1,1) == 0
            log_cursor(1,2) = log_cursor(1,2) + 1; %increment column
            log_cursor(1,1) = 1;
        end
    end

    %determine alarm types
    log_cursor = [1,1];
    for i = 1:size(alarm.title,2) %column
        for j = 1:dim_mal(1) %row
            la = alarm.value(j,i) < l(1,alarm.index(1,i));
            ha = alarm.value(j,i) > h(1,alarm.index(1,i));
            if la == true
                alarm.type(log_cursor(1),log_cursor(2)) = -1;            
            elseif ha == true
                alarm.type(log_cursor(1),log_cursor(2)) = 1;
            else
                alarm.type(log_cursor(1),log_cursor(2)) = 0;
            end
            log_cursor(1) = log_cursor(1) + 1;
        end
        log_cursor(1,2) = log_cursor(2) + 1; %increment column
        log_cursor(1) = 1; %reset row location
    end

    %%plot 
    plot_num = size(alarm.title,2);
    time = trim_mal_num(:,1);
    y = alarm.value;
    for i = 1:(floor(plot_num/8)+1)
        if number == 1
            figure;
        end
        for j = 1:8
            if (i-1)*8+j > plot_num
                if j==1
                    close;
                end
                break;
            end
            h_thre = zeros(dim_mal(1),1) + avg(1,alarm.index(1,(i-1)*8+j))*1.1;
            l_thre = zeros(dim_mal(1),1) + avg(1,alarm.index(1,(i-1)*8+j))*0.9;
            if number == 0
                continue;
            end
            subplot(6,4,j+8*floor(j/5));
                hold on;
                plot(time,y(:,(i-1)*8+j));
                plot(time,h_thre,'--r');
                plot(time,l_thre,'--b');
                title(alarm.title((i-1)*8+j));
                hold off;
            subplot(6,4,j+4+8*floor(j/5));
                atl = alarm.type(:,(i-1)*8+j);
                atl(atl~=-1)=0;
                stairs(time,atl);
                area(time,atl.*-1,'FaceColor','b');
            subplot(6,4,j+8+8*floor(j/5));
                ath = alarm.type(:,(i-1)*8+j);
                ath(ath~=1)=0;
                stairs(time,ath);
                area(time,ath,'FaceColor','r');
        end
    end

    %%output to excel

    report_num = floor(time(end))+1;
    sample_per_h = 360;
    report_name = sprintf('report of %s',filename);
    report_name = replace(report_name,'csv','xlsx');
    report_body = cell(plot_num+2,2*report_num);

    for i = 1: report_num
        report_body(3:plot_num+2,1+2*(i-1)) = cellstr(alarm.title(:));
        for j = 1:plot_num
            if alarm.type(1+360*(i-1),j) == 0
                report_value = '0';
            elseif alarm.type(1+360*(i-1),j) == 1
                report_value = 'H';
            else
                report_value = 'L';
            end
            report_body(2+j,2+2*(i-1)) = cellstr(report_value);
        end
        report_body(1,1+2*(i-1)) = cellstr(sprintf('%d hour',i-1));
        report_body(2,1+2*(i-1)) = cellstr('alarms');
        report_body(2,2+2*(i-1)) = cellstr('values');
    end


    xlswrite(report_name,report_body);
    y = handles;
end




