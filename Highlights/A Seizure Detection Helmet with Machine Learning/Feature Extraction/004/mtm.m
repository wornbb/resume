% input: 
%        data: double
% output:
%         y:double
function y = mtm(data,interval,tappers)
    sum_power = zeros(tappers,24);
    signal_power = (data(:,:)).^2;
    signal_power = signal_power./256;
    for i = 1:tappers
        if i ~= tappers
            sum_power(i,:) = sum(signal_power(1 + interval*(i-1):interval*i,:));
        else
            sum_power(i,:) = sum(signal_power(1 + interval*(i-1):end,:));
            [l,~] = size(signal_power(1 + interval*(i-1):end,:));
        end
    end
    sum_power(1:tappers-1,:) = sum_power(1:tappers-1,:)/interval;
    sum_power(tappers,:) = sum_power(tappers,:)/l ; 
    sum_power = abs(sum_power);
    y = 10*log10(sum_power);
%     y = array2table(sum_power);
%     y.Properties.VariableNames = data.Properties.VariableNames;
end