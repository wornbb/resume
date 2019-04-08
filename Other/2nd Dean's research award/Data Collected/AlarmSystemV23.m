
% number = modes (1 = plot, 0 = no plot)
% 
function [alarm_out time_out avg_out report_out] = AlarmSystemV23(number,mal)
    
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
                xlabel('time (h)')
                ylabel('alarm variable')
                hold off;
            subplot(6,4,j+4+8*floor(j/5));
                atl = alarm.type(:,(i-1)*8+j);
                atl(atl~=-1)=0;
                atl = atl.*-1;
                atl = remove_chattering(atl);
                area(time,atl,'FaceColor','b');
                xlabel('time (h)')
                ylabel('low alarm')
                title(strcat('low alarm for',{' '},alarm.title((i-1)*8+j)));
            subplot(6,4,j+8+8*floor(j/5));
                ath = alarm.type(:,(i-1)*8+j);
                ath(ath~=1)=0;
                ath = remove_chattering(ath);
                area(time,ath,'FaceColor','r');
                xlabel('time (h)')
                ylabel('high alarm')
                title(strcat('high alarm for',{' '},alarm.title((i-1)*8+j)));
        end
    end

    
    
    %%output to excel

    report_num = floor(time(end))+1;
    sample_per_h = 360;
    report_name = sprintf('report of %s',filename);
    report_name = replace(report_name,'csv','xlsx');
    report_body = cell(plot_num*2 + 2,1 + report_num);

    for i = 1: report_num
        for j = 1:plot_num
            report_body((j-1)*2+2,1) = cellstr(strcat(alarm.title(j),'.low'));
            report_body((j-1)*2+3,1) = cellstr(strcat(alarm.title(j),'.high'));
            report_body((j-1)*2+2,i+1) = {'0'};
            report_body((j-1)*2+3,i+1) = {'0'};
            if alarm.type(1+360*(i-1),j) == 0
                %report_value = '0';
            elseif alarm.type(1+360*(i-1),j) == 1
                %report_value = 'H';
                report_body((j-1)*2+3,i+1)={'1'};
            else
                %report_value = 'L';
                report_body((j-1)*2+2,i+1)={'1'};
            end
           % report_body(2+j,2+2*(i-1)) = cellstr(report_value);
        end
        report_body(1,1+i) = cellstr(sprintf('%d hour',i-1));
        report_body(1,1) = cellstr('sensors');
%         report_body(2,1+i) = cellstr('alarms');
    end
    xlswrite(report_name,report_body);
    alarm_out = alarm;
    time_out = time;
    avg_out = avg;
    report_out = report_body;
end

function output = remove_chattering(input)
    output = input;
    
    zero_window_start = 0;
    zero_window_end =0;
    zero_window = 0;
    one_window_start = 0;
    one_windows_end = 0;
    one_window = 0;
    
    change = 0;
    alarm = 0;
    
    [length nothing] = size(input);
    for i = 1:length
        if i == 1
            alarm = input(1,1);
        else
            if input(i,1) ~= input(i-1,1)
                change = 1;
            else
            change = 0;
            end
        end 
        
        if change == 1 && zero_window == 0 && one_window == 0
            if input(i,1) == 1
                one_window_start = i;
                one_window = one_window + 1;
            else
                zero_window_start = i;
                zero_window = zero_window + 1;
            end
        elseif change == 0 && zero_window ~= 0
            zero_window = zero_window + 1;
        elseif change == 0 && one_window ~= 0
            one_window = one_window + 1;
        elseif change == 1 && zero_window ~= 0
            zero_window_end = zero_window_start + zero_window;
            if zero_window < 8 
                for j = zero_window_start:zero_window_end
                    output(j,1) = 1;
                end
            end
            zero_window_start = 0;
            zero_window = 0;
            zero_window_end = 0;
        elseif change == 1 && one_window ~= 0
            one_window_end = one_window_start + one_window;
            if one_window < 8
                for j = one_window_start:one_window_end
                    output(j,1) = 0;
                end
            end
            one_window_start = 0;
            one_window = 0;
            one_window_end = 0;
        end
        
    end
    
    for i = 1:length
        if i ~= 1 && i~=length
            a = input(i-1,1);
            b = input(i,1);
            c = input(i+1,1);
            if a~=b && b~=c
                input(i,1)=c;
            end
        end
    end

end
%             if input(i,1)==1
%                 if one_window_start == 0
%                     ;
%                 else
%                     one_window = one_window + 1;
%                 end
%             else 
%                 if zero_window_start == 0
%                     zero_window_start = 1;
%                     zero_window = zero_winsow + 1;
%                 else
%                     zero_window = zero_winsow + 1;
%                 end
%             end
%             if one_window >= 400 && alarm == 0
%                 alarm = 1;
%             elseif zero_window >= 400 && alarm == 1
%                 alarm = 0;
%             elseif one_window < 400 && alarm == 0
% 
%             end