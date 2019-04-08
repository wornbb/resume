function edf2gui(edf_file_path,gui_dir)
    working_dir = pwd;
    [edf_dir,edf_file_name,edf_ext] = fileparts(edf_file_path);
    edf_name = [edf_file_name,edf_ext];
    gui_name = [edf_file_name,'.txt'];
    cd(edf_dir);
    [~,eeg_data] = edfread(edf_name);
    eeg_data = eeg_data';
    [col,~] = size(eeg_data);
    cd(gui_dir);
    gui_file = fopen(gui_name,'wt');
    fprintf(gui_file,'%%OpenBCI Raw EEG Data\n');
    fprintf(gui_file,'%%Sample Rate = 250.0 Hz\n');
    fprintf(gui_file,'%%First Column = SampleIndex\n');
    fprintf(gui_file,'%%Other Columns = EEG data in microvolts followed by Accel Data (in G) interleaved with Aux Data\n');
    
    place_holder = zeros(col,3);
    index = (1:col)';
    morphed_data = data_morph(eeg_data);
    gui_data = [index morphed_data place_holder];
    fprintf(gui_file,'%d, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f\n',gui_data');
    fclose(gui_file);
    cd(working_dir);
end