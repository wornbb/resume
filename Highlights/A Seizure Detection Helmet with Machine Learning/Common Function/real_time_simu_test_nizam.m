
function output = real_time_simu_test_nizam(offline_data)
    % configuration
    min_f = 1;
    max_f = 24;
    interval_f = 3;
    %....................
    tappers = round((max_f-min_f)/interval_f);

    buffer = zeros(256*2,24);
    
    temp = offline_data;
    % data_morph for nizam data
%     morphed_data(:,1) = temp(:,10) - temp(:,13);
%     morphed_data(:,2) = temp(:,11) - temp(:,8);
%     morphed_data(:,3) = temp(:,8) - temp(:,21);
%     morphed_data(:,4) = temp(:,21) - temp(:,14);
%     
%     morphed_data(:,5) = temp(:,11) - temp(:,14);
%     morphed_data(:,6) = temp(:,13) - temp(:,20);
%     morphed_data(:,7) = temp(:,20) - temp(:,7);
%     morphed_data(:,8) = temp(:,7) - temp(:,10);
    morphed_data = offline_data;
    morphed_data = clean(morphed_data,'online');
    [l,~] = size(morphed_data);
    t = fix(l/8);
    for i  = 1:(t)
        data = morphed_data(1+8*(i-1):8*(i),:);
        % manipulate the channel set up so that the channel set up mirrors
        % the real set up.
        % real_data = data_morph(data);
        old_buffer = buffer;
        buffer = update_queue(old_buffer,8);
        buffer(1:8,17:24) = data(:,:);
        if i>64*3
            output = [output;buffer];
        elseif i == 64*3
            output = buffer;
        end
        
    end
end