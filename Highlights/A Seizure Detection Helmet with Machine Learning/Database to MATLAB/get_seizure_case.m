function [title number] = get_seizure_case
    % Initialize GUI
    prompt = 'Directory of the Database Goes Here';
    dlg_title = 'Input Directory';
    % Get Dir
    data_dir = inputdlg(prompt,dlg_title);
    current_dir = pwd;
    cd(char(data_dir));
    % Search .seizures
    meta_objective = dir('*.seizures');
    % Output results
    title = meta_objective;
    number = size(meta_objective);
    cd(current_dir);
end

