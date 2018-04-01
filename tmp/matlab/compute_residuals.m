function all_residuals = compute_residuals(Ps, us, U)
U=[U;1];
all_residuals=[];
for i=1:length(Ps)
    all_residuals=[all_residuals;(Ps{i}(1,:)*U)/(Ps{i}(3,:)*U)-us(1,i);(Ps{i}(2,:)*U)/(Ps{i}(3,:)*U)-us(2,i)];      
end
