% assuming sample freq = 256
% input: double, double,double,double
% output: double, double array
function  [y x] = find_freq(data,min_freq,max_freq)
    [l,~] = size(data);
    Fs = 256;
    T = 1/Fs;
    total_freq = Fs*(0:(l/2))/l;
    f_sta = find(abs((total_freq - min_freq)) <= Fs/l);
    f_end = find(abs(total_freq - max_freq) <= Fs/l);
    freq_range = total_freq(f_sta:f_end);
    x = freq_range;
    y = data(f_sta:f_end,:);
end