function normalized_signal = normalize_signal(signal)
    [row,col] = size(signal);
    % RMS normalization
    N = row*col/8;
    amp = abs(signal);
    p_amp = amp.^2;
    sum_amp = sum(p_amp);
    overlap_per_channel = sum_amp(1,1:8)+sum_amp(1,9:16)+sum_amp(1,17:24);
    RMS = sqrt(overlap_per_channel/N);
    normalized_signal = signal./[RMS RMS RMS];
end