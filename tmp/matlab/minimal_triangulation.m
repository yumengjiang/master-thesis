function U = minimal_triangulation(Ps, us)
M=[];
b=[];
n=length(Ps);
for i=1:n
    M=[M;Ps{i}(:,1:3),zeros(3,i-1),-[us(:,i);1],zeros(3,n-i)];
    b=[b;-Ps{i}(:,4)];
end
theta=M\b;
U=zeros(3,1);
U=theta(1:3);
    