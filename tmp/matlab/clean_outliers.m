function removed_indices = clean_outliers(U)
% Remove outliers, only keep those U with all coordinates less than 20
minvals = quantile(U,0.01,2);
maxvals = quantile(U,0.99,2);

removed_indices = U(1,:) > maxvals(1,:) | U(1,:) < minvals(1,:);
for kk = 2:3
    removed_indices = removed_indices | U(kk,:) > maxvals(kk,:) | U(kk,:) < minvals(kk,:);
end

for i=1:size(U,2)
    if abs(U(1,i))>20 || abs(U(2,i))>20 || abs(U(3,i))>20
        removed_indices(i)=1;
    end
end