function eye_detection(handles,varargin)
    working_dir = pwd;
    % opne GUI
%     t = timer;
%     t.StartDelay = 2;
%     t.TimerFcn = @open_gui;
%     start(t);
    open_gui;
        
    % start TCP/IP server
    host = tcpip('127.0.0.1', 5204, 'NetworkRole', 'server','INPUT',1024*10);
    fopen(host);
    cd(working_dir);
    setappdata(handles.Parent,'eye_run',1);
    setappdata(handles.Parent,'sei_run',0);
    run = getappdata(handles.Parent,'eye_run');
        % stop TCP/IP server
    % This script is an interrupt, handling potential unfinished work in
    % the previous routine. 
    echotcpip('off')
    delete(instrfind);

    
    %filter/bucket matrixes = test matrixes for placing stored data in %
    filtermatrix = zeros(50000,2);
    filtermatrix2 = zeros(50000,2);
    bucket_matrix = zeros(50,2);
    %Fnorm = 30hz filter, Fnorm2 = 5hz filter
    Fnorm = 30/(250/2);
    Fnorm2 = 2/(250/2);
    blinktic = 0;
    blinktoc = 0;
    blink_signal = 0;
    blink_signal2 = 0;
    %
    df = designfilt('lowpassfir','FilterOrder',10,'CutoffFrequency',Fnorm);
    df2 = designfilt('lowpassfir','FilterOrder',10,'CutoffFrequency',Fnorm2);
    %delete(instrfind);
    %below is code for arduino, commented at this time for testing purposes.
        %  arduino = serial('COM6','BaudRate',9600);
    %fopen(arduino); 
    %t = fscanf(arduino)
    %ignore variables below
    time3 = 1:512;
    i_count = 0;
% initializing 
    min_f = 1;
    max_f = 24;
    type = 1;
    stop = 0;
    channels = 8;
    incomplete = 0;
    fs = 256;
    window_speed = 8; % default window speed. it may (unlikely) increase a bit
  
    % processign window. Feature will be extrated from here
    buffer_window = zeros(fs*2,channels*3);
    buffer_filter = buffer_window;
    

    
    % initialize the first 6s window buffer
    
% real time processing

% fill the empty 6s window for processing 
% if the first 6s window is not filled, the feature extraction algorithm
% will not work properly.
    while buffer_window(1,1) == 0
        [package,incomplete] = get_package(host,incomplete);
        [window_speed,~] = size(package);
        buffer_window = update_window(buffer_window,window_speed);
        buffer_window(1:window_speed,17:24) = package(:,:);
        drawnow;
        run = getappdata(handles.Parent,'sei_run');
    end

    while run
        % terminate process by typing s
        [package,incomplete] = get_package(host,incomplete);
        [window_speed,~] = size(package);
        buffer_window = update_window(buffer_window,window_speed);
        buffer_window(1:window_speed,17:24) = package(:,:);
        filtered_window3 = [buffer_window;buffer_window]; 
        filtered_window3 = filtered_window3 .*0.0223; % convert to nV from uV
        filtered_window3 = filtered_window3(513:1024,:);
        filtered_window4 = [filtered_window3(:,17:24);filtered_window3(:,9:16)];
       %below: filtered window4 filtering
     for j = 1:8    
      filtered_window4(:,j) = (filtered_window4(:,j));
        %5hz filter = df2   
        filtered_window4(:,j) = filter(df2,filtered_window4(:,j));
        filtered_window4(1:15,j) = filtered_window4(16,j);
        %take the average, and then subtract it from each value
        avg_val = mean(filtered_window4(:,j));
        filtered_window4(:,j) = filtered_window4(:,j) - avg_val;
        filtered_window4(1:15,j) = 0;
     end
%load the bucket_matrix
        bucket_matrix = [bucket_matrix;filtered_window4(:,4) filtered_window4(:,8)];
        i_count = i_count + 1;
        
        negative = 1;
        figure(1);
        %below: chan56 = left - right channels
        
        chan56 = filtered_window4(:,1) - filtered_window4(:,5);
        chan56(1:20) = 0;
        % this program below is the new left right implementation, and 
        % needs some more refining. Plots are displayed to show peaks.
        timechan56 = 1:400;
        [pkl,locl] = findpeaks(chan56(1:400),timechan56,'MinPeakDistance',60,'MinPeakWidth',20,'MinPeakHeight',5);
        findpeaks(chan56(1:400),timechan56,'MinPeakDistance',60,'MinPeakWidth',20,'MinPeakHeight',5);
        drawnow;
        figure(2);
        timechan56 = 1:400;
        [pkr,locr] = findpeaks(chan56(1:400)*-1,timechan56,'MinPeakDistance',60,'MinPeakWidth',20,'MinPeakHeight',5);
        findpeaks(chan56(1:400)*-1,timechan56,'MinPeakDistance',60,'MinPeakWidth',20,'MinPeakHeight',5);
        drawnow;
        figure(3);
        test99 = 1:3;
        obj99 = [1 1 1];
        if(length(pkr) > 1 && length(pkr) < 4)
            disp('right');
            obj99 = obj99*3;
        end
        if(length(pkl) > 1 && length(pkl) < 4)
            disp('left');
            obj99 = obj99*-3;
        end
        plot(obj99);
        drawnow;
%leftright_ratio = ratio of chan56 compared to left and right channel mean.
      leftright_ratio = zeros(length(chan56(:,1)),1);
      
      leftright_ratio2 = zeros(length(chan56(:,1)),1);
      %computer left right ratio
        for iii = 1:length(chan56(:,1))
            
%         leftright_ratio(iii,1) = chan56(iii,1)/filtered_window4(iii,5);
        leftright_ratio2(iii,1) = chan56(iii,1)/(mean(abs(filtered_window4(iii,1)) + abs(filtered_window4(iii,5))));
        end
%leftright_ratio3 = fliplr leftright_ratio2
        leftright_ratio3 = fliplr(leftright_ratio2);
        opr = 0;
        opl = 0;
%blinkr and blinkl = channels 4 8 = channels above left and right eye.
        blinkr = filtered_window4(:,4);
        blinkl = filtered_window4(:,8);
        blinkr_det = 0;
        blinkl_det = 0;
        blink_det = 0;

        for i = 5:350
            opblinkr1 = blinkr(i+40) - blinkr(i);
            opblinkr2 = blinkr(i+40) - blinkr(i+70);
            if(opblinkr1 < -0.25 && opblinkr1 > -0.8 && opblinkr2 < -0.25 && opblinkr2 > -0.8)
                blinkr_det = blinkr_det + 1;
            end
            opblinkl1 = blinkl(i+40) - blinkl(i);
            opblinkl2 = blinkl(i+40) - blinkl(i+70);
            if(opblinkl1 < -0.25 && opblinkl1 > -0.8 && opblinkl2 < -0.25 && opblinkl2 > -0.8)
                blinkl_det = blinkl_det + 1;
            end
        end
        if(blinkr_det > 5 && blinkl_det > 5)
            blink_signal = 10;
            
            blinktic = cputime;
            blinktoc = cputime;
            
                
           % fprintf(arduino,'7');
           
        end
       
        if(blink_signal == 10)
            blinktoc = cputime;
            opz = blinktoc - blinktic;
            if(opz > 1)
                blink_signal2 = 0;
            end
            if(opz > 5)
                blink_signal = 0;
            end
        end
        opl = 0;
        opr = 0;
        % this is the old left right detection implementation.
        %constants need to be changed due to the new conversion factor
        for i = 1:350
            opleftright_ratio = leftright_ratio3(i);
            opright = 0;
            if(abs(opleftright_ratio) < 1.6)
            opright = leftright_ratio3(i) - leftright_ratio3(i+50);
            if(opright < -0.5)
                opr = opr + 1;
            end
            if(opright > 0.5)
                opl = opl + 1;
            end
            end
        end
        
        
        S55 = [opl, opr, blinkr_det, blinkl_det];
%         disp(S55);
        if(opl > 30 && blink_signal == 10)
            fopen(dev);
            fprintf(dev,'r');
            fclose(dev);
        end
        if(opr > 30 && opl < 30 && blink_signal == 10)
            fopen(dev);
            fprintf(dev,'r');  
            fclose(dev);
        end
        chan78 = filtered_window3(:,15) - filtered_window3(:,16);
        kk = 1;
        if(kk == 0)
        end
        drawnow;
        run = getappdata(handles.Parent,'eye_run');
        [~,~,key]=KbCheck(-1);      
        if any(KbName(key)=='s')||any(KbName(key)=='S')
            run = 0;
        end
    end


    % stop TCP/IP server
    fclose(host);
    KbQueueStop;
    echotcpip('off')
    delete(instrfind);
end