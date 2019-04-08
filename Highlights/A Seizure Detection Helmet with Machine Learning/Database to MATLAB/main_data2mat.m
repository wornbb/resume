function main_data2mat(start)
    % keywords 
    script_dir = 'C:\Users\shen1\Desktop\develoopment\Database to MATLAB';
    data_dir = 'D:\Compressed\physionet';
    seizure_dir = 'D:\Compressed\physionet\Seizure';
    non_dir = 'D:\Compressed\physionet\NonSeizure';
    cd(data_dir);
    folder_name = ls('*chb*'); % char list
    [number_folder,~] = size(folder_name); % = 24
    sample_rate = 256; % Hz
%     seizure_guide = 
%   struct with fields:
%          test_name: [40×12 char]
%          next_stop: [40×1 double]
%     number_seizure: [40×1 double]
%         start_time: [40×10 double]
%           end_time: [40×10 double]
%            channel: {40×38 cell}
 
    % import all channels
    for i = start:number_folder
        cd(strcat(data_dir,'\',folder_name(i,:)));;
        % get seizure_guide which tells when the seizure happens for each
        % test
        seizure_guide = get_seizure_guide;
        [number_test,number_channel] = size(seizure_guide.channel);
        % import all edf
        for j = 1:number_test
            
            [hdr,record] = edfread(seizure_guide.test_name(j,:));
            [non_duplicate,non_duplicate_index,~] = unique(hdr.label);
            empty_index = cellfun(@isempty,non_duplicate);
            data = record(non_duplicate_index(~empty_index),:)';
            non_duplicate = non_duplicate(~empty_index);
            [~,edf_name,~] = fileparts(seizure_guide.test_name(j,:));
            if seizure_guide.number_seizure(j,1) == 0         
                non_seizure = array2table(data,'VariableNames',non_duplicate); 
                save_name = sprintf('NonSeizure_%s',edf_name);
                cd(non_dir);
                save(save_name,'non_seizure');
                cd(strcat(data_dir,'\',folder_name(i,:)));
                
            % for the case that seizure presents
            else
                pre_end_time = 1;
                for k = 1:seizure_guide.number_seizure(j,1)
                    % the data was stored in reverse order
                    index = seizure_guide.number_seizure(j,1) - k + 1;
                    start_time = seizure_guide.start_time(j,index) * sample_rate;
                    end_time = seizure_guide.end_time(j,index) * sample_rate;
                    seizure_data = data(start_time:end_time,:);
                    seizure = array2table(seizure_data,'VariableNames',non_duplicate);
                    % save
                    save_name = sprintf('HaveSeizure_%s_%d',edf_name,k);
                    cd(seizure_dir)
                    save(save_name,'seizure');
                    cd(strcat(data_dir,'\',folder_name(i,:)));;
                    %...................
                    non_seizure_data = data(pre_end_time:start_time,:);
                    non_seizure = array2table(non_seizure_data,'VariableNames',non_duplicate);
                    save_name = sprintf('NonSeizure_%s_%d',edf_name,k);
                    cd(non_dir);
                    save(save_name,'non_seizure');
                    cd(strcat(data_dir,'\',folder_name(i,:)));;
                    if k == seizure_guide.number_seizure(j,1)
                        non_seizure_data = data(end_time:end,:);
                        non_seizure = array2table(non_seizure_data,'VariableNames',non_duplicate);
                        save_name = sprintf('NonSeizure_%s_%d',edf_name,k+1);
                        cd(non_dir);
                        save(save_name,'non_seizure');
                        cd(strcat(data_dir,'\',folder_name(i,:)));;
                    end
                    pre_end_time = end_time;
                    
                end
         
            end
        end
        
        
        
    end


    cd(script_dir);
end