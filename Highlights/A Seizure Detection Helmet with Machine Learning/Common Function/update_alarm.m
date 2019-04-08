% alarm is in a column
function updated_alarm = update_alarm(prev_alarm)
    updated_alarm = prev_alarm;
    updated_alarm(1:end-8,1) = prev_alarm(9:end,1);
end