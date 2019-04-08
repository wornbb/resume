function estimate_time(index,total_iterat,start_time)
    current_time = clock();
    if index == 10
        time = current_time - start_time;
        estimate_time = total_iterat/index*(time(4)*3600+time(5)*60+time(6));
        
        fprintf('10 iteration report: %f\n',estimate_time);
    elseif index == 100
        time = current_time - start_time;
        estimate_time = total_iterat/index*(time(4)*3600+time(5)*60+time(6));
        
        fprintf('100 iteration report: %f\n',estimate_time);
    elseif index/round(total_iterat/10)==floor(index/round(total_iterat/10))
        percent = index/round(total_iterat/10)*10;
        time = current_time - start_time;
        estimate_time = total_iterat/index*(time(4)*3600+time(5)*60+time(6));   
        fprintf('%d Percent report: %f\n',percent,estimate_time);
    end


end