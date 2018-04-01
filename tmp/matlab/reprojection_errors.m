function errors = reprojection_errors(Ps, us, U)
U=[U;1];
for i=1:length(Ps)
    if Ps{i}(3,:)*U<0
        errors(i)=Inf;
    else
        eus(1,i)=(Ps{i}(1,:)*U)/(Ps{i}(3,:)*U);
        eus(2,i)=(Ps{i}(2,:)*U)/(Ps{i}(3,:)*U);
        errors(i)=sqrt((eus(1,i)-us(1,i))^2+(eus(2,i)-us(2,i))^2);        
    end
end
