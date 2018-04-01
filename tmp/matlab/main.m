clc;
clear;
warning('off');
addpath(genpath('sift'));

%% 2.1 Reconstruction from two views
% Initilize
for i=1:13
    name{i}=['../../result/',num2str(i),'.png'];
end
num1=1;
num2=num1+1;
img_ori1=read_image(name{num1});
img_ori1=imresize(img_ori1,0.5);
img_ori2=read_image(name{num2});
img_ori2=imresize(img_ori2,0.5);
img1=rgb2gray(img_ori1);
img2=rgb2gray(img_ori2);
K=[350.6847, 0, 332.4661;
    0, 350.0606, 163.7461;
    0, 0, 1];
threshold=10;
loop=50;

% Match SIFT features of the first two views
[coords1, d1] = extractSIFT(img1);
[coords2, d2] = extractSIFT(img2);
index12=matchFeatures(d1',d2','MaxRatio',0.9,'MatchThreshold',100);
u1=round(coords1(:,index12(:,1)));
u2=round(coords2(:,index12(:,2)));
for i=1:size(index12,1)
    ucolor(:,i)=img_ori1(u1(2,i),u1(1,i),:);
end

% Use RANSAC to estimate the best P2 with most inliers
Pc1=[diag([1,1,1]),zeros(3,1)];
P1=K*Pc1;
nbr_inliers=0;
for n=1:loop
    % Randomly select 5 u pairs to estimate Pc2
    subset=randperm(size(u1,2),5);
    Pc2 = five_point_algorithm(u1(:,subset), u2(:,subset), K, K);  
    % For each possible Pc2, calculate reprojection errors and count inliers 
    for m=1:length(Pc2)
        tP2=K*Pc2{m};
        Ps={P1,tP2};
        errors=[];
        for i=1:size(u1,2)
            us=[u1(:,i),u2(:,i)];
            Uhat = ransac_triangulation(Ps, us, threshold);
            errors= [errors,reprojection_errors(Ps, us, Uhat)];
        end       
        % Mark P2 with the most inliers
        tnbr_inliers=sum(errors<=threshold);
        if tnbr_inliers>nbr_inliers
            nbr_inliers=tnbr_inliers;
            P2=tP2;
        end
    end
end
% Mark all useful information 
messages(num2).round=num2;
messages(num1).nbr_Us=size(u1,2);
messages(num2).inliers=nbr_inliers;
messages(num2).inlier_rate=nbr_inliers/size(u1,2)/2;
center=-inv(P2(:,1:3))*P2(:,4);
messages(num2).center=center;
disp(['round: ',num2str(num1),', nU: ',num2str(size(u1,2)),', inliers: ',num2str(nbr_inliers),', inlier_rate: ',num2str(nbr_inliers/size(u1,2)/2),', center: ',num2str(center')]);

%Estimate U w.r.t view 1 and view 2
Ps={P1,P2};
k=0;
for i=1:size(u1,2)
    us=[u1(:,i),u2(:,i)];
    Uhat = ransac_triangulation(Ps, us, threshold);
    errors=reprojection_errors(Ps, us, Uhat);
    % Only keep inliers
    if sum(errors<=threshold)==length(errors)       
        k=k+1;
        U(:,k) = refine_triangulation(Ps, us, Uhat);
        ids(k)=i;
    end
end
% Remove outliers
removed_indices=clean_outliers(U);
Uc=U(:,~removed_indices);

%% 2.2 Book-keeping
track=[];
f2f_track=[];
% Only track inliers
ids_new=ids(~removed_indices);
for i=1:size(Uc,2)
    track(i).image_ids=[num1,num2];
    track(i).feature_ids=index12(ids_new(i),:);
    track(i).feature=[u1(:,ids_new(i)),u2(:,ids_new(i))];
    track(i).triangulated_point=Uc(:,i);
    track(i).color=ucolor(:,ids_new(i));
    f2f_track{num1}(index12(ids_new(i),1))=i;
    f2f_track{num2}(index12(ids_new(i),2))=i;
end
P={P1,P2};

%% 2.3 Adding a new image
for i=2:19
    [f2f_track,track,P,messages]=triangulations(i,i+1,name,f2f_track,track,P,messages);
end

%% 3D reconstruction
for i=1:length(track)
    U_new(:,i)=track(i).triangulated_point;
    Ucolor_new(i,:)=track(i).color;
end
[Uc_new,removed_indices_new2]=clean_for_plot(U_new);
color_new=Ucolor_new(~removed_indices_new2,:);
figure;
hold on;
scatter3(Uc_new(1,:),Uc_new(2,:),Uc_new(3,:),[],color_new,'.');
for i=1:length(P)
    C_new=-inv(P{i}(:,1:3))*P{i}(:,4);
    scatter3(C_new(1),C_new(2),C_new(3),'ro');
    text(C_new(1),C_new(2),C_new(3),num2str(i));
end










