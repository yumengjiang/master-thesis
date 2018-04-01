function positive = check_depths(Ps, U)
for i=1:length(Ps)
    lambda(i)=Ps{i}(3,:)*[U;1];
end
positive=lambda>0;