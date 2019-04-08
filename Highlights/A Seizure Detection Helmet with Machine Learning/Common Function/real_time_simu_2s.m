function feature_vector = real_time_simu_2s(offline_data,fx,type,filter_bank)
    % configuration
    min_f = 1;
    max_f = 24;
    interval_f = 3;
    %....................
    tappers = round((max_f-min_f)/interval_f);
    [l,~] = size(offline_data);
    t = fix(l/512);
    buffer = zeros(512,24);
    start_time = clock();
    for i  = 1:(t)
        data = offline_data{1+512*(i-1):512*(i),:};
        % manipulate the channel set up so that the channel set up mirrors
        % the real set up.
        real_data = data_morph(data);
        old_buffer = buffer;
        buffer = update_queue(old_buffer,512);
        buffer(1:512,17:24) = real_data(:,:);
        if i>=3
        partial_feature_vector = fx(buffer,type,min_f,max_f,filter_bank);
            if i == 3
                feature_vector = partial_feature_vector;
            end
        feature_vector = [feature_vector;partial_feature_vector];
        end
        estimate_time(i,(t),start_time);
    end
end
