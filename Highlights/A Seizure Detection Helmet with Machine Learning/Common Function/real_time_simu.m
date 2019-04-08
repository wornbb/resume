function feature_vector = real_time_simu(offline_data,fx,type,filter_bank)
    % configuration
    min_f = 1;
    max_f = 24;
    interval_f = 3;
    %....................
    tappers = round((max_f-min_f)/interval_f);
    [l,~] = size(offline_data);
    t = fix(l/8);
    buffer = zeros(256*2,24);
    for i  = 1:(t)
        data = offline_data{1+8*(i-1):8*(i),:};
        % manipulate the channel set up so that the channel set up mirrors
        % the real set up.
        real_data = data_morph(data);
        old_buffer = buffer;
        buffer = update_queue(old_buffer,8);
        buffer(1:8,17:24) = real_data(:,:);
        if i>=64*3
        partial_feature_vector = fx(buffer,type,min_f,max_f);
            if i == 64*3
                feature_vector = partial_feature_vector;
            end
        feature_vector = [feature_vector;partial_feature_vector];
        end
        
    end
end
% function feature_vector = real_time_simu(offline_data,fx,type)
%     % configuration
%     min_f = 1;
%     max_f = 24;
%     interval_f = 3;
%     %....................
%     tappers = round((max_f-min_f)/interval_f);
%     [l,~] = size(offline_data);
%     t = fix(l/8);
%     buffer = zeros(256*2,24);
%     for i  = 1:(t)
%         data = offline_data{1+8*(i-1):8*(i),:};
%         % manipulate the channel set up so that the channel set up mirrors
%         % the real set up.
%         real_data = data_morph(data);
%         old_buffer = buffer;
%         buffer = update_queue(old_buffer,8);
%         buffer(1:8,17:24) = real_data(:,:);
%         if i>=64*3
%         partial_feature_vector = fx(buffer,type,min_f,max_f);
%             if i == 64*3
%                 feature_vector = partial_feature_vector;
%             end
%         feature_vector = [feature_vector;partial_feature_vector];
%         end
%         
%     end
% end