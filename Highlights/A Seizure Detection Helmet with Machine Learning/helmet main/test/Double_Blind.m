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

% data = non_seizure{:,:};
% data = seizure{:,:};
%data = data';
cd('C:\Users\shen1\Desktop\develoopment\DD_test');
files = ls;
for file_index = 3:7
    cd('C:\Users\shen1\Desktop\develoopment\DD_test');
    files = ls;
    load(files(file_index,:));
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
    [~,length] = size(data);
    j = 1;
    for i = 1:2:length
        reduced_data(:,j) = data(:,i);
        j = j + 1;
    end
%cd('C:\Users\shen1\Desktop\develoopment\MATLAB_Seizure_Detection\helmet main\test');
%load('test_data.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lines below is only needed if handling unprepared data
data = reduced_data';


% mannual morph Nizam data
morphed_data = DD_morph(hdr,data);
% prepared_data = real_time_simu_test_nizam(morphed_data);
[row,~] = size(morphed_data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot_alarm =zeros(256*10000,1);
start_time = clock();
chattering_index = 0;
iterate = row-256;
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
            feature_vector = fx005(buffer_window,type,min_f,max_f);
    end
            %%%%%%%%%%%%%%%%%
            feature_vector = feature_vector(:,1:24);
            feature_vector = feature_morph(feature_vector);
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
            %[chattering_index,alarm] = chattering_removal(chattering_index,alarm);
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
            title(files(file_index,:))
            hold on;
            plot(morphed_data(:,7),'b');
            hold off;
end
