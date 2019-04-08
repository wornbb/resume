
function y = fx003(buffer,type,min_f,max_f)

    tappers = fix((max_f-min_f)/0.86);
    % FFT
    data_fft = varfun(@fft,buffer);
    [data_fft, freq_range] = find_freq(data_fft,min_f,max_f);
    % mtm
    interval = fix(size(freq_range)/tappers); % how many data points in window
    table_sei = mtm(data_fft,interval(2),tappers);
    
    % eign in freq domain
    current_fft = data_fft(:,17:24);
    for i = 1:tappers
        mean_f = mean(current_fft{1+interval(2)*(i-1):interval(2)*i,:});
        var_f  = var(current_fft{1+interval(2)*(i-1):interval(2)*i,:});
        normalized = (current_fft{1+interval(2)*(i-1):interval(2)*i,:}-mean_f)./var_f;
        cor = corrcoef(normalized);
        e = eig(cor);
        freq_feature(i,:) = e';
    end
    
    % eign in time domain
    data = buffer(:,17:24);
    for i = 1:tappers
        mean_t = mean(data{1+interval(2)*(i-1):interval(2)*i,:});
        var_t  = var(data{1+interval(2)*(i-1):interval(2)*i,:});
        normalized = (data{1+interval(2)*(i-1):interval(2)*i,:}-mean_t)./var_t;
        cor = corrcoef(normalized);
        e = eig(cor);
        time_feature(i,:) = e';
    end
    % classify
    f_domain = array2table(freq_feature);
    t_domain = array2table(time_feature);
    table_sei = [table_sei f_domain t_domain];
    if iscategorical(type)
        table_sei.type(:,1) = type;
    end
    y = table_sei;

end