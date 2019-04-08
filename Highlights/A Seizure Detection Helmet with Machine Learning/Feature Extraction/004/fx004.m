% input 
%     buffer: double 
% output
%     y: table
function y = fx004(buffer,type,min_f,max_f)

    tappers = fix((max_f-min_f)/1);
    % FFT
    data_fft = fft(buffer);
    %%%%data_fft = varfun(@fft,buffer);
    [data_fft, freq_range] = find_freq(data_fft,min_f,max_f);
    % mtm
    interval = fix(size(freq_range)/tappers); % how many data points in window
    energy = mtm(data_fft,interval(2),tappers);
    
    % eign in freq domain
    current_fft = data_fft(:,17:24);
    for i = 1:tappers
        mean_f = mean(current_fft(1+interval(2)*(i-1):interval(2)*i,1:8));
        var_f  = var(current_fft(1+interval(2)*(i-1):interval(2)*i,1:8));
        normalized = (current_fft(1+interval(2)*(i-1):interval(2)*i,1:8)-mean_f)./var_f;
        cor = corrcoef(normalized);
        e = eig(cor);
        freq_feature(i,:) = e';
    end
    
    % eign in time domain
    data = buffer(:,17:24);
    for i = 1:tappers
        mean_t = mean(data(1+interval(2)*(i-1):interval(2)*i,1:8));
        var_t  = var(data(1+interval(2)*(i-1):interval(2)*i,1:8));
        normalized = (data(1+interval(2)*(i-1):interval(2)*i,1:8)-mean_t)./var_t;
        cor = corrcoef(normalized);
        e = eig(cor);
        time_feature(i,:) = e';
    end
    % classify
    feature_array = [energy freq_feature time_feature];
    feature_table = array2table(feature_array);
    if iscategorical(type)
        feature_table.type(:,1) = type;
    end
    y = feature_table;

end