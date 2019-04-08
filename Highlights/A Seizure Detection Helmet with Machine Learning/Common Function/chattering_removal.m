function [chattering_index,alarm] = chattering_removal(index,pre_alarm)
    num_latency = 3;%15
    alarm = pre_alarm;
    chattering_index = index;
    if alarm(end,1) ~= alarm(end-256,1)
        if index <= num_latency
            chattering_index = chattering_index + 1;
            alarm(end-255:end,1) = pre_alarm(end-511:end-256,1);
        else
            chattering_index = 0;
            alarm(end-255:end,1) = ~pre_alarm(end-511:end-256,1)*100;
        end
    end
end