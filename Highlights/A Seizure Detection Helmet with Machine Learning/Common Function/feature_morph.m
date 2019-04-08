function feature = feature_morph(incorrect_feature)
    

    [row,col] = size(incorrect_feature);
    iteration = row/7;
    feature = zeros(24*iteration,8);
    for i = 1:iteration
        for k = 1:3
            for j = 1:8
                feature(8*(k-1)+24*(i-1)+j,1:7)=incorrect_feature{1+7*(i-1):7*i,j+8*(k-1)};
                feature(8*(k-1)+24*(i-1)+j,8) = j*100;
            end
        end
    end
    feature = table(feature);
end