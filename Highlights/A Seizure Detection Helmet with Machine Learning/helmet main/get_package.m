function [package,incomplete] = get_package(host,incomplete)
    row = 0;
    while row < 8
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
        [row,~] = size(good_data);
    end
    % for Carson Data
    temp = package;
    package(:,1) = temp(:,1) - temp(:,3);
    package(:,2) = temp(:,3) - temp(:,5);
    package(:,3) = temp(:,7) - temp(:,5);
    package(:,4) = temp(:,1) - temp(:,7);
    
    package(:,5) = temp(:,2) - temp(:,4);
    package(:,6) = temp(:,4) - temp(:,6);
    package(:,7) = temp(:,8) - temp(:,6);
    package(:,8) = temp(:,2) - temp(:,8);

    % for Nizam 
%     temp = package;
%     package(:,1) = temp(:,10) - temp(:,13);
%     package(:,2) = temp(:,11) - temp(:,8);
%     package(:,3) = temp(:,8) - temp(:,21);
%     package(:,4) = temp(:,21) - temp(:,14);
%     
%     package(:,5) = temp(:,11) - temp(:,14);
%     package(:,6) = temp(:,13) - temp(:,20);
%     package(:,7) = temp(:,20) - temp(:,7);
%     package(:,8) = temp(:,7) - temp(:,10);

end