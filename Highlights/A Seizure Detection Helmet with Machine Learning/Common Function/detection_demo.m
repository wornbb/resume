% worker 3
function detection_demo()
    stop = 0;
    while stop == 0
        [~,~,key]=KbCheck(-1);  
        if any(KbName(key)=='s')||any(KbName(key)=='S')
            stop = 1;
        end      
        data_n_results = labReceive(2);
        data = data_n_results.data;
        results = data_n_results.results;
        index = results == 'seizure';
        seizure_where = find(index);
        
        [~,n] = size(results);
        [l,~] = size(data);
        [~,N] = size(seizure_where);
        interval_length = fix(l/n);
        
        time = (1:interval_length*n)*(2/(interval_length*n));
        plot(time,data(1:interval_length*n,1),'b');
        y_value = ylim;
        hold on;
        for i = 1:N
            rectangle('Position',[time(1,1+interval_length*(seizure_where(1,i)-1)),...
                                    y_value(1,1),...
                                    2/n,...
                                    abs(y_value(1,1))+y_value(1,2)],...
                        'FaceColor','red',...
                        'EdgeColor','red');
                                    
        end
        plot(time,data(1:interval_length*n,1),'b');
        xlim(x_value);
        hold off;
    end
end