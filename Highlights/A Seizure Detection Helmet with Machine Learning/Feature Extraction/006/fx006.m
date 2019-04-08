% input 
%     buffer: double 
% output
%     y: table
% fn = fieldnames(a); % yields cell-array {'e', 'f'}
% a.(fn{1}) % take first field of struct a

function y = fx006(buffer,type,min_f,max_f,filter_bank)
    y = zeros(24,9);
    fn = fieldnames(filter_bank);
    %normalized_signal = normalize_signal(buffer);
    normalized_signal = buffer;
    response = zeros(24,1);
    response = categorical(response);
    feature = zeros(24,9);
    multiply4filter = [normalized_signal;normalized_signal;normalized_signal;normalized_signal];
    for i = 1:8
        filtered_signal = filter(filter_bank.(fn{i}),multiply4filter,1);
        filtered_signal = filtered_signal(512*3+1:end,:);
        signal_power = filtered_signal.^2;
        feature(1:8,i) = signal_power(1,17:24)';
        feature(9:16,i) = signal_power(1,9:16)';
        feature(17:24,i) = signal_power(1,1:8)';
       	feature(i,9) = i*100;
        feature(i+8,9) = i*100;
        feature(i+16,9) = i*100;
    end
    
    y = table(feature);
    if iscategorical(type)
        response(:,1) = type;
        y = [table(feature) table(response)];
    end

%     % configuration
%     interval_f = 3;
%     %..................
%     
%     tappers = fix((max_f-min_f)/interval_f);
%     % FFT
%     data_fft = fft(buffer);
%     %%%%data_fft = varfun(@fft,buffer);
%     [data_fft, freq_range] = find_freq(data_fft,min_f,max_f);
%     % mtm
%     interval = fix(size(freq_range)/tappers); % how many data points in window
%     energy = mtm(data_fft,interval(2),tappers);
%     
%     % classify
%     feature_array = [energy freq_feature time_feature];
%     feature_table = array2table(feature_array);
%     if iscategorical(type)
%         feature_table.type(:,1) = type;
%     end
%     y = feature_table;
 
end
