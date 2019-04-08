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
            if zero_window < 200 
                for j = zero_window_start:zero_window_end
                    output(j,1) = 1;
                end
            end
            zero_window_start = 0;
            zero_window = 0;
            zero_window_end = 0;
        elseif change == 1 && one_window ~= 0
            one_window_end = one_window_start + one_window;
            if one_window < 200
                for j = one_window_start:one_window_end
                    output(j,1) = 0;
                end
            end
            one_window_start = 0;
            one_window = 0;
            one_window_end = 0;
        end
        
    end

end