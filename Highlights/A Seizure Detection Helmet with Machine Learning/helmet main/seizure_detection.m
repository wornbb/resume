function seizure_detection(handles,varargin)
    working_dir = pwd;
    % opne GUI
%     t = timer;
%     t.StartDelay = 3;
%     t.TimerFcn = @open_gui;
%     start(t);
    
    cd(working_dir);
    open_gui;
    % start TCP/IP server
    host = tcpip('127.0.0.1', 5204, 'NetworkRole', 'server','INPUT',1024*10); % buffer size is 10 KB, should be way more than enough
    fopen(host);
%     % stop TCP/IP server
%     % This script is an interrupt, handling potential unfinished work in
%     % the previous routine. 
%     echotcpip('off')
%     delete(instrfind);
% Initialization
    % set up for arduino communication
%     arduino = serial('COM5','BaudRate',9600);
%     fopen(arduino); 
%     t = fscanf(arduino);
    % set enable loop in this script and disable in the other one.
    % the commands here is duplicated to avoid any potential problem
    setappdata(handles.Parent,'eye_run',0);
    setappdata(handles.Parent,'sei_run',1);
    run = getappdata(handles.Parent,'sei_run');
    % a bunch of stuff for signal acquiring and detection
    min_f = 1;
    max_f = 24;
    type = 1; % any number will do here.
    channels = 8;
    incomplete = 0;
    fs = 256;
    window_speed = 256*2; % default window speed. it may (unlikely) increase a bit
    alarm = zeros(2*256,1);
    chattering_index = 0;
    % processign window. Feature will be extrated from here
    buffer_window = zeros(fs*2,channels*3);
    % load classifier
    cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\Model Tranning\temp');
    load final_classifier_005.mat

    

    % initialize the first 6s window buffer
    % real time processing
    % fill the empty 6s window for processing 
    % if the first 6s window is not filled, the feature extraction algorithm
    % will not work properly.
    
    while buffer_window(1,1) == 0 && run

        
        [package,incomplete] = get_package(host,incomplete);
        [window_speed,~] = size(package);
        buffer_window = update_window(buffer_window,window_speed);
        buffer_window(1:window_speed,17:24) = package(:,:);
        drawnow;
        run = getappdata(handles.Parent,'sei_run');
    end

    while run

        [package,incomplete] = get_package(host,incomplete);
        [window_speed,~] = size(package);
        buffer_window = update_window(buffer_window,window_speed);
        buffer_window(1:window_speed,17:24) = package(:,:);
        feature_vector = fx005(buffer_window,type,min_f,max_f);
        feature_vector = feature_morph(feature_vector);
        detection_results = trainedModel.predictFcn(feature_vector);
        judge_index = countcats(detection_results); % judge_index(1,1) = occurence of seizure
        alarm = update_alarm(alarm);
        if judge_index(1,1) >= 12
            judge = 'seizure';
%             fprintf(arduino,'5');
            alarm(end-7:end,1) = ones(8,1)*100;
        else
            judge = 'nonseizure';
            alarm(end-7:end,1) = zeros(8,1);
        end
        [chattering_index,alarm] = chattering_removal(chattering_index,alarm);
        % ploting function, need some change
%         plot(alarm,'r');
%         hold on;
%         plot(fliplr(filtered_window(:,17)),'b');
%         drawnow;
%         hold off;
        data_n_results = struct('data',buffer_window,'results',detection_results);
        fprintf('%s\n',judge);
        drawnow;
        run = getappdata(handles.Parent,'sei_run');
        [~,~,key]=KbCheck(-1);  
        if any(KbName(key)=='s')||any(KbName(key)=='S')
            run = 0;
        end
    end
    dos('taskkill /F /IM javaw.exe');
    % stop TCP/IP server
    fclose(host);
    KbQueueStop;
    echotcpip('off')
    delete(instrfind);
end