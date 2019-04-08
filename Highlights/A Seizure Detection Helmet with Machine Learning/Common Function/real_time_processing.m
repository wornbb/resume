function real_time_processing()
%fill the buffer while keep data clean
%send to feature extraction continuously
%need interupt

% initialization
    min_f = 1;
    max_f = 24;
    type = 1;
    stop = 0;
    channels = 8;
    window_speed = 8;
    fs = 256;
    % A small package of data used to update the processing window
    % increament of the moving window
    buffer_data_package = zeros(window_speed,channels);
    % processign window. Feature will be extrated from here
    buffer_window = zeros(fs*2,channels*3);
    % load classifier
    cd('C:\Users\shen1\Desktop\develoopment\Test\004');
    load matlab.mat

    
% real time processing

% fill the empty 6s window for processing 
% if the first 6s window is not filled, the feature extraction algorithm
% will not work properly.
%     h = dialog('Position',[300 300 750 450],'Name','Initialization Message');
%     txt = uicontrol('Parent',h,...
%             'Style','text',...
%             'Position',[25 100 700 300],...
%             'FontSize',60,...
%             'String','Initializing');
    for i = 1:6
%         txt.String = sprintf('Program will start in %d seconds',7 - i);
        fprintf('Program will start in %d seconds',7 - i);
        for j = 1:32*2*3
            buffer_data_package = labReceive(1);
            buffer_window = update_queue(buffer_window,window_speed);
            buffer_window(1:8,17:24) = buffer_data_package(:,:);
        end
    end
%     txt.String = sprintf('Start detecting seizures, press "S" or "s" to stop');
%     btn = uicontrol('Parent',h,...
%        'Position',[225 10 300 80],...
%        'String','Get It',...
%        'FontSize',30,...
%        'Callback','delete(gcf)');
% continuous processing

    while stop == 0
        [~,~,key]=KbCheck(-1);  
        if any(KbName(key)=='s')||any(KbName(key)=='S')
            stop = 1;
        end      
        buffer_data_package = labReceive(1);
        buffer_window = update_queue(buffer_window,window_speed);
        buffer_window(1:8,17:24) = buffer_data_package(:,:);
        feature_vector = fx005(buffer_window,type,min_f,max_f);
        detection_results = trainedClassifier.predictFcn(feature_vector);
        data_n_results = struct('data',buffer_window,'results',detection_results);
        fprintf('%s',detection_reults);
        labSend(data_n_results,3);
    end
    

end