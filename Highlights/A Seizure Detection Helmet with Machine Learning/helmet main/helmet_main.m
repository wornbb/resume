function helmet_main()
    working_dir = pwd;
    path('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\toolbox',path);
    path('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\helmet main',path);
    path('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\Common Function',path);
    path('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\Common Function',path);
    temp_f = fopen('install_log.txt','r');
    install_index = fread(temp_f);
    if install_index == 48
        fclose(temp_f);
        temp_f = fopen('install_log.txt','w+');
        fprintf(temp_f,'1');
        cd('..\toolbox\Psychtoolbox');
        run('SetupPsychtoolbox.m');
        d = dialog('Position',[300 300 250 150],'Name','Installation Instruction');

        txt = uicontrol('Parent',d,...
                   'Style','text',...
                   'Position',[20 80 210 40],...
                   'String','Go to Command Window and Follow the instructions');

        btn = uicontrol('Parent',d,...
                   'Position',[85 20 70 25],...
                   'String','Close',...
                   'Callback','delete(gcf)');
        cd(working_dir);
    end
    fclose(temp_f);
    
    
    
    r=instrhwinfo('Bluetooth');
    indicator_cell = strfind(r.RemoteNames,'itead');
    index = cell2mat(indicator_cell);
    dn1 = cell2mat(r.RemoteNames(index));
    dev = Bluetooth(dn1,1);
    fopen(dev);
    fwrite(dev,'s');
    fclose(dev);
%     
    
%     %initializing 
%     min_f = 1;
%     max_f = 24;
%     type = 1;
%     stop = 0;
%     channels = 8;
%     incomplete = 0;
%     fs = 256;
%     window_speed = 8; % default window speed. it may (unlikely) increase a bit
%   
%     % processign window. Feature will be extrated from here
%     buffer_window = zeros(fs*2,channels*3);
%     % load classifier
%     %cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\Test\004');
%     load trainedClassifier.mat
%     
%     % start TCP/IP server
%     host = tcpip('127.0.0.1', 5204, 'NetworkRole', 'server','INPUT',1024*10);
%     fopen(host);
%     alarm = zeros(2*256,1);
%     chattering_index = 0;
%     % initialize the first 6s window buffer
%     
% % real time processing
% 
% % fill the empty 6s window for processing 
% % if the first 6s window is not filled, the feature extraction algorithm
% % will not work properly.
%     while buffer_window(1,1) == 0
%         [package,incomplete] = get_package(host,incomplete);
%         [window_speed,~] = size(package);
%         buffer_window = update_window(buffer_window,window_speed);
%         buffer_window(1:window_speed,17:24) = package(:,:);
%     end
% 
% b=[ 2.001387256580675e-001, 0.0, -4.002774513161350e-001, 0.0, 2.001387256580675e-001];
% a = [1.0, -2.355934631131582e+000, 1.941257088655214e+000, -7.847063755334187e-001, 1.999076052968340e-001];
% an=[1.000000000000000e+000, -2.467782611297853e-001, 1.944171784691352e+000, -2.381583792217435e-001, 9.313816821269039e-001];
% bn = [9.650809863447347e-001, -2.424683201757643e-001, 1.945391494128786e+000, -2.424683201757643e-001, 9.650809863447347e-001];
% % set up for arduino communication
% 
% arduino = serial('COM5','BaudRate',9600);
% fopen(arduino); 
% t = fscanf(arduino);
f = figure('Visible','on','Position',[360,500,500,400]);
setappdata(f,'sei_run',1);
setappdata(f,'eye_run',1);
hseizure = uicontrol(f,'Style','pushbutton','String','Seizure Detection',...
    'Position',[100,250,300,100],... 
    'Fonts',20,...
    'Interruptible','on',...
    'Callback',@seizure_detection);

heye = uicontrol(f,'Style','pushbutton','String','Eye Movement',...
    'Position',[100,50,300,100],...
    'Fonts',20,...
    'Interruptible','on',...
    'Callback',{@eye_detection,dev});
%     while stop == 0
%         % terminate process by typing s
%         [~,~,key]=KbCheck(-1);  
%         if any(KbName(key)=='s')||any(KbName(key)=='S')
%             stop = 1;
%         end      
%         tic
%         [package,incomplete] = get_package(host,incomplete);
%         [window_speed,~] = size(package);
%         buffer_window = update_window(buffer_window,window_speed);
%         buffer_window(1:window_speed,17:24) = package(:,:);
%         % Since we only need to play prepared seizure files, no need for
%         % filtering
% % 
% %         filtered_window = [buffer_window;buffer_window]; 
% %         filtered_window = filter(b,a,filtered_window); % bandpass filter 1-50 Hz
% %         filtered_window = filter(bn,an,filtered_window); % notch filter 60 Hz
% %         filtered_window = filtered_window ./ 1000; % convert to nV from uV
% %         filtered_window = filtered_window(513:1024,:);
%         
%         feature_vector = fx005(buffer_window,type,min_f,max_f);
%         detection_results = trainedClassifier.predictFcn(feature_vector);
%         judge_index = countcats(detection_results); % judge_index(1,1) = occurence of seizure
%         alarm = update_alarm(alarm);
%         if judge_index(1,1) >= 3
%             judge = 'seizure';
% %             fprintf(arduino,'5');
%             alarm(end-7:end,1) = ones(8,1)*100;
%         else
%             judge = 'nonseizure';
%             alarm(end-7:end,1) = zeros(8,1);
%         end
%         [chattering_index,alarm] = chattering_removal(chattering_index,alarm);
%         plot(alarm,'r');
%         hold on;
%         plot(fliplr(filtered_window(:,17)),'b');
%         drawnow;
%         hold off;
%         data_n_results = struct('data',buffer_window,'results',detection_results);
%         %fprintf('%s\n',judge);
%     end
%     toc
%     % stop TCP/IP server
%     fclose(host);
%     kbQueueStop;
%     echotcpip('off')
%     delete(instrfind);
end


