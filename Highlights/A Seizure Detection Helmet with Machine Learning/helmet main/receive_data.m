% get a 8channel 32 ms data package 
function [package,incomplete] = receive_data(host,prev_incomplete)
    clean_start = 0;
    incomplete = 0;
    % check if it is a clean start, no unit package is separated between
    % this and previous data receive process
    if ~isstring(prev_incomplete)
        clean_start = 1;
    end
    % The first data inquiry *special 
    % The beginning of the data might be incomplete due to previoius
    % transfer
    % The end of the data might be incomplete
    % The data is can also be multiple of 1*8 unit data transfer 
    while host.BytesAvailable<8
    end
    data = fread(host, host.BytesAvailable);
    
    str = string(char(data')); % the data is in ascii code, transform into char
    words = str.split; % in str, all data are in one string. split it 
    
    count_e = count(words,'e'); % # of received unit package 
    e_place = find(count_e); % locate the place of each unit package
    first_e = e_place(1,1);
    last_e = e_place(end,1);
    
    % pre processing of dirty start 
    if clean_start == 0
        % check the first data point received
        % consider a random float 7.123 
        % during the transmission, it can be divided like 7. in the
        % prev_incomplete and 123 in the current first element
        % define 7. as head, and 123 as body, just like a beheaded person
        [~,l] = size(words(1,1));
        if l < 8 && first_e ~= 1
            body = words(1,1);
            head = prev_incomplete(end,1);
            resurrected = head + body;
        end
        % now concate the first unit package
        try
            first_unit = [prev_incomplete(1:end-1,1);resurrected;words(1:first_e-1,1)];
        catch
            first_unit = [prev_incomplete(1:end,1);words(1:first_e-1,1)];
        end
    end
    
    % processing of the rest data
    if strcmp(words(end,1),string(''))
        words = words(1:end-1,1);
    end
    [length,~] = size(words); % get how many data we received
    % check the completeness of last unit package.
    if length - last_e > 0
        % general case
        incomplete = words(last_e + 1:end,1);
        clean_data = words(first_e + 1:last_e,1);
        % special case
        if first_e == last_e
            clean_data = string('');
        end
    elseif length - last_e == 0
        % general case 
        clean_data = words(first_e + 1:end,1);
        % special case 
        if first_e == last_e && clean_start == 0
            clean_data = string('');
        elseif first_e == last_e && clean_start == 1
            clean_data = words(1:last_e,1);
        end
    else
        error('something went wrong with then client _from last unit check');
    end
    
  
    if strcmp(clean_data,string(''))
        package = string('empty');    
    elseif last_e == first_e
        for i = 1:first_e
            row = floor(i/9) + 1;
            col = mod(i,9);
            if col ~= 0
                package(row,col) = clean_data(i,1);
            end
        end 
    else
        for i = 1:last_e - first_e
            row = floor(i/9) + 1;
            col = mod(i,9);
            if col ~= 0
                package(row,col) = clean_data(i,1);
            end
        end 
    end

end