function [U, nbr_inliers] = ransac_triangulation(Ps, us, threshold)
U=zeros(3,1);
nbr_inliers=0;

for i=1:size(Ps,2)*(size(Ps,2)-1)/2
    index = randperm(size(Ps,2),2);
    tPs={Ps{index(1)},Ps{index(2)}};
    tU = minimal_triangulation(tPs, us(:,index));
    errors= reprojection_errors(Ps, us, tU);
    
    tnbr_inliers=sum(errors<=threshold);
    if tnbr_inliers>nbr_inliers
        nbr_inliers=tnbr_inliers;
        U=tU;
    end
end
