% worker 1
function get_data()
    stop = 0;
    cd('C:\Users\shen1\Desktop\develoopment\Test')
    [~,data] = edfread('Nov10_2016_EEGrecording_normaltoseizure_spikewave_frozenseizure.edf');
    data = data';
    [length,~] = size(data);
    index = 1;
    while stop == 0
        [~,~,key]=KbCheck(-1);  
        if any(KbName(key)=='s')||any(KbName(key)=='S')
            stop = 1;
        end      
        buffer = zeros(8,8);
% load 
        for i = 1:8
            temp_data = data(index,2:9);
            index = index + 1;
            buffer(i,:) = temp_data;
        end
% clean and reload
        buffer = clean(buffer,'online');
        [l,~] = size(buffer);
        while l ~= 8
            for j = l+1:8
                %disp('pass');
                temp_data = data(index,2:9);
                index = index + 1;
                buffer(j,:) = temp_data;
            end
            buffer = clean(buffer,'online');
            [l,~] = size(buffer);
            if index >= length-8
                stop = 1;
                break;
            end
        end

        if index == length
            stop = 1;
        end
        labSend(buffer,2);
    end
end

%%
%     h = dialog('Position',[300 300 750 450],'Name','Initialization Message');
%     txt = uicontrol('Parent',h,...
%             'Style','text',...
%             'Position',[25 100 700 300],...
%             'FontSize',[60],...
%             'String','Initializing');
%     for i = 1:6
%         txt.String = sprintf('Program will start in %d seconds',7 - i);
%         pause(1)
%     end
%     txt.String = sprintf('Start detecting seizures, press "S" or "s" to stop');
%     btn = uicontrol('Parent',h,...
%        'Position',[225 10 300 80],...
%        'String','Get It',...
%        'FontSize',[30],...
%        'Callback','delete(gcf)');