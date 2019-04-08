format shortg
scale_uv_per_count = 4.5 / ((2^23-1)) / 24  * 1000000; % confirmed from experiments
package_size = 74;
stop = 0;
% start tcpip server
host = tcpip('127.0.0.1', 5204, 'NetworkRole', 'server','INPUT',1024*10);
fopen(host);

fprintf('loaded');
i = 1;
incomplete = 0;
while stop == 0
    [package,incomplete] = receive_data(host,incomplete);
    if strcmp(package,string('empty'))
    else
        try
            package = double(package);
            good_data = [good_data;package];
        catch
            package = double(package);
            good_data = package;
        end
    end

%     try
        
%         data = fread(t, t.BytesAvailable);
%         
%         current_time = clock;
%         future_time = clock;
% %         while future_time(1,6) - current_time(1,6) <= 0.51
% %             future_time = clock;
% %         end
% %         fprintf('%d\n',data);
%         str = string(char(data'));
%         words = str.split;
%         temp = double(words(1,1));
%         if isnan(temp)
%             temp = 0;
%         end
%         num(1,i) = temp;
%         i = i+1;
%         plot(num);
%     catch
%      end
end
fclose(host);
echotcpip('off')