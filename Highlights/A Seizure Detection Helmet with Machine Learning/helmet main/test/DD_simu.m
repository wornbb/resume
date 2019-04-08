
function output = DD_simu(offline_data)
    % configuration
    min_f = 1;
    max_f = 24;
    interval_f = 3;
    %....................
    tappers = round((max_f-min_f)/interval_f);

    buffer = zeros(256*2,24);
    
    temp = offline_data;
    temp = clean(temp,'online');
    [l,~] = size(temp);
    t = fix(l/8);
    for i  = 1:(t)
        data = temp(1+8*(i-1):8*(i),:);
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