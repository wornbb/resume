% mode = online 
%      = offline
function output = clean(input,mode)
    ind = strcmp(mode,'online');
    if ind
        buffer = input;
    else
        buffer = input{:,:};
    end
    % discard duplicate measurements
    [l,~] = size(buffer);
    prev = buffer(1,:);
    for i = 2:l
        difference = buffer(i,:)-prev;
        prev = buffer(i,:);
        if_no_zero = 1;
        [~,channels]=size(difference);
        for j = 1:channels-1
            if_no_zero = if_no_zero&difference(1,j)&difference(1,j+1);
        end
        if ~if_no_zero
            buffer(i,1) = NaN;
        end
    end
    [l,~] = size(buffer);
    % error 
    if l<= 4
        errordlg('at least 4 measurements are missing, serious connection problem');
    end
    
    % output
    pre_output = array2table(buffer);
    output = rmmissing(pre_output);
    if ind
        output = table2array(output);
    end
end