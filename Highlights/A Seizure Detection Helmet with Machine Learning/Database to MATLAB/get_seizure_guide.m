% this function could only be called in the working_dir that defined int
% the main_data2mat.m

% this function access a particular file (summary) in the working_dir, which contains
% the info for each test indicating,for each EEG test, "if","how many","when","how long" a
% seizure happened.

% it return a structure array, for each element in the array, the element
% carries the seizure info of the corresponding test.


function seizure_guide = get_seizure_guide
    % initialize
    working_dir = pwd;
    [~,folder_name] = fileparts(working_dir);
    all_test = ls('*.edf');
    [number_test,~]= size(all_test);
    seizure_guide = struct(...
        'test_name',all_test,...
        'next_stop',zeros(number_test,1),...
        'number_seizure',zeros(number_test,1),...
        'start_time',zeros(number_test,10),...
        'end_time',zeros(number_test,10),...
        'channel',{cell(number_test,1)}...
    );
    str_file_name = 'File Name: ';
    str_number_seizure = 'Number of Seizures in File: ';
    str_channel = 'Channel ';
    str_channel_indicator = '****';
    str_file = 'File';
    str_start_time = 'Start Time: ';
    str_end_time = 'End Time: ';
    % get the name the particualr file (summary)
    summary_name = sprintf('%s-summary.txt',folder_name);
    
    % open file
    summary = fopen(summary_name);
    % process
    % when the file name is detected, this loop start extracting the
    % seizure time related to that.
    current_number_seizure = 0;
    current_stop = 1;
    channel_holder = {cell(1,50)};
    
    
    tline = fgetl(summary);
    while ischar(tline)
        % the following 3 adjacent if blocks are used to 
        %   1. identify which test is reading now
        %   2. store the channel information to eatch test
        % the reverse_words method has a lower efficiency than strfind.
        % hence not used universally.
        if contains(tline,str_channel_indicator)
           channel_indicator = 1;
        end
        if contains(tline,str_channel)
           new_line = reverse_words(tline);
           channel_name = sscanf(new_line,'%s',1);
           channel_holder(1,channel_indicator) = {channel_name};
           channel_indicator = channel_indicator + 1;
        end
        if contains(tline,str_file_name)
            name_place = strfind(tline,str_file_name);
            current_test_name = sscanf(tline(name_place + length(str_file_name):end),'%s',1);
            result_in_cell = strfind(string(seizure_guide.test_name),current_test_name);
            current_index = find(~cellfun('isempty', result_in_cell));
            [~,channel_size] = size(channel_holder);
            seizure_guide.channel(current_index,1:channel_size) = channel_holder(1,:);
        end
        
        % following code determines all necessary seizure signal information for each test
        if contains(tline,str_number_seizure)
            number_place = strfind(tline,str_number_seizure);
            current_number_seizure = sscanf(tline(number_place+length(str_number_seizure):end),'%d',1);
            seizure_guide.number_seizure(current_index,1) = current_number_seizure;
            if current_number_seizure ~= 0
                seizure_guide.next_stop(current_stop,1) = current_index;
                current_stop = current_index;
            end
        end
        if ~contains(tline,str_file)
            if contains(tline,str_start_time)
                start_place = strfind(tline,str_start_time);
                current_start_time = sscanf(tline(start_place + length(str_start_time):end),'%d',1);
                seizure_guide.start_time(current_index,current_number_seizure) = current_start_time;
            end
            if contains(tline,str_end_time)
                end_place = strfind(tline,str_end_time);
                current_end_time = sscanf(tline(end_place + length(str_end_time):end),'%d',1);
                seizure_guide.end_time(current_index,current_number_seizure) = current_end_time;
                current_number_seizure = current_number_seizure - 1;
            end
        end
        if feof(summary)
             seizure_guide.next_stop(current_stop,1) = 1994811;
        end
        tline = fgetl(summary);
    end
    
    fclose(summary);
end























%function [ seizure_start_time_offset_in_seconds, seizure_length_in_seconds ] = get_seizure_period( annotation_file_location )
     
%     
%     byte_array = fread(file_descriptor);
%     file_descriptor = fopen(annotation_file_location);
%     seizure_start_time_offset_in_seconds = bin2dec(strcat(dec2bin(byte_array(39)),dec2bin(byte_array(42))));
%     
%     seizure_length_in_seconds = byte_array(50);

% end


