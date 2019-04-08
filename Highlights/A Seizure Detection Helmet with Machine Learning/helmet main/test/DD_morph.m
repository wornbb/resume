function morphed_data = DD_morph(hdr,data)
    channel = hdr.label;
    channel = upper(string(channel));
    
    FP1 = strfind(channel,'FP1');
    FP1 = find(~cellfun(@isempty,FP1));

    O1 = strfind(channel,'O1');
    O1 = find(~cellfun(@isempty,O1));
    
    FP2 = strfind(channel,'FP2');
    FP2 = find(~cellfun(@isempty,FP2));
    
    F8 = strfind(channel,'F8');
    F8 = find(~cellfun(@isempty,F8));
    
    T6 = strfind(channel,'T6');
    T6 = find(~cellfun(@isempty,T6));
    
    O2 = strfind(channel,'O2');
    O2 = find(~cellfun(@isempty,O2));
    
    T5 = strfind(channel,'T5');
    T5 = find(~cellfun(@isempty,T5));
    
    F7 = strfind(channel,'F7');
    F7 = find(~cellfun(@isempty,F7));

    output(:,1) = data(:,FP1) - data(:,O1);
    output(:,2) = data(:,FP2) - data(:,F8);
    output(:,3) = data(:,F8) - data(:,T6);
    output(:,4) = data(:,T6) - data(:,O2);
    
    output(:,5) = data(:,FP2) - data(:,O2);
    output(:,6) = data(:,O1) - data(:,T6);
    output(:,7) = data(:,T5) - data(:,F7);
    output(:,8) = data(:,F7) - data(:,FP1);

    morphed_data = output;





end