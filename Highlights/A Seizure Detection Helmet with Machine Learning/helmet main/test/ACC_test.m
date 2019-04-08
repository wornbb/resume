% %% Nizam (remember change get_package.m)
% % Load Nizam Data. 
% cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\Test');
% % [hdr data] = edfread('Nov10_2016_EEGrecording_normaltoseizure_spikewave_frozenseizure.edf');
% [hdr data] = edfread('chb20_12.edf');
% % Same data as above, but already prepared
% % cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\helmet main\test');
% % load('test_data.mat');
% % cd('C:\Users\shen1\Desktop\develoopment\Seizure_Double_Blind');
% % files = ls;

% cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection');
% load('trainedClassifier.mat');
% cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\Model Tranning\temp');
% load('trainClassifier_energy.mat');
 %data = non_seizure{:,:};
cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\Feature Extraction\006');
load('filter_bank.mat');
 data = seizure{:,:};
%data = data';

    min_f = 1;
    max_f = 24;
    type = 1;
    stop = 0;
    channels = 8;
    incomplete = 0;
    fs = 256;
    window_speed = 256;
    buffer_window = zeros(fs*2,channels*3);
    alarm = zeros(2*256,1);
    
    [data_row,~] = size(data);
morphed_data=zeros(data_row,8);
%cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\helmet main\test');
%load('test_data.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lines below is only needed if handling unprepared data
% data = data';
% manually morph database data
morphed_data(:,1) = data(:,9).* -1;

% F7 -  T5(P7) = F7T7 + T7P7
morphed_data(:,2) = data(:,6) + data(:,21);

% P7 - O1
morphed_data(:,3) = data(:,17);

% FP1 - O1 (another route) = FP1F3 + F3C3+C3P3+P3O1
morphed_data(:,4) = data(:,8) + data(:,4) + data(:,1) + data(:,15);

% FP2 - F8
morphed_data(:,5) = data(:,11).* -1;

% F8 - T6(P8) = F8T8 + T8P8 
morphed_data(:,6) = data(:,7) + data(:,22);

% T6 - O2
morphed_data(:,7) = data(:,19);

% FP2 - O2 (another route) = FP2F4 + F4C4 + C4P4 + P4O2
morphed_data(:,8) = data(:,10) + data(:,5) + data(:,2) + data(:,16);


% prepared_data = real_time_simu_test_nizam(morphed_data);
[row,~] = size(morphed_data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot_alarm =zeros(data_row,1);
start_time = clock();
chattering_index = 0;
iterate = row*0.9-8;
    for i = 1:256:iterate
        estimate_time((i-1)/256,floor(iterate/256),start_time);

        % initialize the first 6s window buffer

        if buffer_window(1,1) == 0
            package = morphed_data(i:i+255,:);
            %[window_speed,~] = size(package);
            buffer_window = update_window(buffer_window,window_speed);
            buffer_window(1:window_speed,17:24) = package(:,:);

        else
            % terminate process by typing s
            package = morphed_data(i:i+255,:);
            buffer_window = update_window(buffer_window,window_speed);
            buffer_window(1:window_speed,17:24) = package(:,:);
    %         filtered_window = [buffer_window;buffer_window]; 
    %         filtered_window = filter(b,a,filtered_window); % bandpass filter 1-50 Hz
    %         filtered_window = filter(bn,an,filtered_window); % notch filter 60 Hz
    %         filtered_window = filtered_window ./ 1000; % convert to nV from uV
    %         filtered_window = filtered_window(513:1024,:);
    %         feature_vector = fx005(filtered_window,type,min_f,max_f);
    try
           % feature_vector = fx005(buffer_window,type,min_f,max_f);
           feature_vector = fx006(buffer_window,type,min_f,max_f,filter_bank);
    end
            %%%%%%%%%%%%%%%%%
            %feature_vector = feature_vector(:,1:24);
            %feature_vector = feature_morph(feature_vector);
            detection_results = trainedModel.predictFcn(feature_vector);
            judge_index = countcats(detection_results); % judge_index(1,1) = occurence of seizure
            alarm = update_alarm(alarm);
            if judge_index(1,1) >= 12
                judge = 'seizure';
    %             fprintf(arduino,'5');
                alarm(end-255:end,1) = ones(256,1)*100;
            else
                judge = 'nonseizure';
                alarm(end-255:end,1) = zeros(256,1);
            end
            [chattering_index,alarm] = chattering_removal(chattering_index,alarm);
            plot_alarm(i:i+255,1) = alarm(end-255:end,1);
    %         plot(alarm,'r');
    %         hold on;
    %         plot(fliplr(buffer_window(:,17)),'b');
    %         drawnow;
    %         hold off;
    %         data_n_results = struct('data',buffer_window,'results',detection_results);
              %fprintf('%s\n',judge);
        end

    end
            figure;
            plot(plot_alarm(1:row,1),'r');
            hold on;
            plot(morphed_data(:,7),'b');
            hold off;
beep;
        %data_n_results = struct('data',buffer_window,'results',detection_results);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure; 
% plot_data = 0;
% a = 1;
% b = 1;
% % delete(instrfind);
% % arduino = serial('COM5','BaudRate',9600);
% 
% % fopen(arduino); 
% % t = fscanf(arduino);
% alarm = zeros(30*256,1);
% chattering_index = 0;
% plot_alarm = [];
% plot_data = [];
% for i = 1:8:row
%     filtered_window = prepared_data(512*(i-1)+1:512*i,:);
%     feature_vector = fx005(filtered_window,type,min_f,max_f);
%     detection_results = trainedClassifier.predictFcn(feature_vector);
%     %[time,~] = size(seizure);
%     [r,~] = size(plot_data);
%     if r <= 30*256
%         plot_data = [plot_data;fliplr(filtered_window(:,1));fliplr(filtered_window(:,9));fliplr(filtered_window(:,17))];
%     else
%         plot_data(1:end-8,1) = plot_data(9:end,1);
%         plot_data(end-7:end,1) = fliplr(filtered_window(1:8,17));
%     end
% 
%     judge_index = countcats(detection_results);
%     
%     alarm = update_alarm(alarm);
%     if judge_index(1,1) >= 3
% 
%         alarm(end-7:end,1) = ones(8,1)*100;
% 
% 
%         %fprintf(arduino,'5');
%     else
% 
%         alarm(end-7:end,1) = zeros(8,1);
%  
%     end
%     [chattering_index,alarm] = chattering_removal(chattering_index,alarm);
%     plot_alarm = [plot_alarm;alarm(end-7:end,1)];
%     
% %     plot(plot_alarm,'r');
% %     %hold on;
% %     plot(plot_data,'b');
%     %hold off;
%     drawnow;
% end
% [sample_points,~]=size(plot_alarm);
% time = (1:sample_points)/fs;
% figure;
% hold on;
% for i = 1:8
%     plot(morphed_data(:,i));
% end
% hold off;
% figure;
% plot(time,plot_alarm*0.09);