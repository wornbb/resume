
[row,col] = size(Tfft);
iteration = row/7;
start_time = clock();
feature = zeros(24*iteration,8);
response = zeros(24*iteration,1);
response = categorical(response);
for i = 1:iteration
    for k = 1:3
        for j = 1:8
            feature(8*(k-1)+24*(i-1)+j,1:7)=Tfft{1+7*(i-1):7*i,j+8*(k-1)}';
            feature(8*(k-1)+24*(i-1)+j,8) = j*100;
            response(8*(k-1)+24*(i-1)+j,1) = Tfft{1+7*(i-1),41};
        end
    end
    estimate_time(i,iteration,start_time);
end
a = table(feature);
b = table(response);
correct_train = [a b];