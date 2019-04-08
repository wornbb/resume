function real_data = data_morph(data)
% T3, T4, T5 and T6 — as T7, T8, P7 and P8 respectively.

% the data input should have the following channel set up (same order)
% Columns 1 through 7
% 
%     'C3P3'    'C4P4'    'CZPZ'    'F3C3'    'F4C4'    'F7T7'    'F8T8'
% 
%   Columns 8 through 13
% 
%     'FP1F3'    'FP1F7'    'FP2F4'    'FP2F8'    'FT10T8'    'FT9FT10'
% 
%   Columns 14 through 20
% 
%     'FZCZ'    'P3O1'    'P4O2'    'P7O1'    'P7T7'    'P8O2'    'T7FT9'
% 
%   Columns 21 through 22
% 
%     'T7P7'    'T8P8'
[row,~] = size(data);
real_data = zeros(row,8);
% FP1 - F7
real_data(:,1) = data(:,9).* -1;

% F7 -  T5(P7) = F7T7 + T7P7
real_data(:,2) = data(:,6) + data(:,21);

% P7 - O1
real_data(:,3) = data(:,17);

% FP1 - O1 (another route) = FP1F3 + F3C3+C3P3+P3O1
real_data(:,4) = data(:,8) + data(:,4) + data(:,1) + data(:,15);

% FP2 - F8
real_data(:,5) = data(:,11).* -1;

% F8 - T6(P8) = F8T8 + T8P8 
real_data(:,6) = data(:,7) + data(:,22);

% T6 - O2
real_data(:,7) = data(:,19);

% FP2 - O2 (another route) = FP2F4 + F4C4 + C4P4 + P4O2
real_data(:,8) = data(:,10) + data(:,5) + data(:,2) + data(:,16);

end 