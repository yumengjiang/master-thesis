function [f2f_track,track,P,messages]=triangulations(num1,num2,name,f2f_track,track,P,messages)
%% Match SIFT features to last reconstructed view
%Initilize
img_ori1=read_image(name{num1});
img_ori2=read_image(name{num2});
img1=rgb2gray(img_ori1);
img2=rgb2gray(img_ori2);
[row,col]=size(img1);
px=col/2;
py=row/2;
width=5.72/max(row,col);
K=inv([width,0,-width*px;
    0,width,-width*py;
    0,0,5.8]);
threshold=10+num1/2;% Threshold increases with the number of new view 

% Compute SIFT features for the new view and only match them to last reconstructed view
[coords1, d1] = extractSIFT(img1);
[coords2, d2] = extractSIFT(img2);
index=matchFeatures(d1',d2','Method','Approximate','MaxRatio',0.9,'MatchThreshold',100);
u1=round(coords1(:,index(:,1)));
u2=round(coords2(:,index(:,2)));
ucolor=[];
for i=1:size(index,1)
    ucolor(:,i)=img_ori1(u1(2,i),u1(1,i),:);
end

%% Update track
nbr_track=length(track);
k=nbr_track;
for i=1:size(u1,2)
    flag=false;
    % Match new features to existing feature tracks and update them
    for j=1:nbr_track
        if track(j).image_ids(end)==num1 && track(j).feature_ids(:,end)==index(i,1)         
            flag=true;
            track(j).image_ids=[track(j).image_ids,num2];
            track(j).feature_ids=[track(j).feature_ids,index(i,2)];
            track(j).feature=[track(j).feature,u2(:,i)];
        end
    end
    % If fail to match new features to all existing feature tracks, create a new feature track
    if flag==false
        k=k+1;
        track(k).image_ids=[num1,num2];
        track(k).feature_ids=index(i,:);
        track(k).feature=[u1(:,i),u2(:,i)];
        track(k).triangulated_point=[Inf;Inf;Inf];
        track(k).color=ucolor(:,i);
        f2f_track{num1}(index(i,1))=k;
        f2f_track{num2}(index(i,2))=k;
    end
end

%% 2.4 Camera pose estimation
% Find out those feature groups with new features and at least three members match to each other
k=0;
Us=[]; 
us3=[];
us=[];
for i=1:length(track)
    if track(i).image_ids(end)==num2 && length(track(i).image_ids)>2 
        k=k+1;
        l(k)=length(track(i).image_ids);
        Us(:,k)=track(i).triangulated_point;
        us3{k}=track(i).feature;
        us(:,k)=track(i).feature(:,end);
    end
end
nU=size(Us,2);

% Use the feature point to feature track correspondences to estimate the pose of the new camera
P3_inliers=0;
inlier_rate=messages(end).inlier_rate;
for n=1:100/inlier_rate^3
    % Randomly select 3 u and U pairs to estimate Pc3
    subset=randperm(nU,3);
    Pc3 = minimal_camera_pose(us(:,subset), Us(:,subset), K);
    % For each possible Pc3, calculate reprojection errors and count inliers 
    for m=1:length(Pc3)
        tP3=K*Pc3{m};  
        tcenter=-inv(tP3(:,1:3))*tP3(:,4);
        lcenter=-inv(P{end}(:,1:3))*P{end}(:,4);
        % A good new pose should be near to last camera pose, this step makes it much faster and more reliable, but you should have knowledge about the new pose 
%         if sum((tcenter-lcenter)<[2;0.5;1] & (tcenter-lcenter)>[-2;-0.5;-0.3])==3
        if sum((tcenter-lcenter)<[3;3;3] & (tcenter-lcenter)>[-3;-3;-1])==3
            errors=[];
            % Use all feature groups to calculate reprojection errors and count inliers         
            for i=1:nU
                Ps={P{end-l(i)+2:end},tP3};
                Uhat = ransac_triangulation(Ps, us3{i}, threshold);
                % Bigger feature groups can have more inliers, in other words, a good pose can gain more inliers from big feature groups than bad pose
                errors= [errors,reprojection_errors(Ps, us3{i}, Uhat)];
            end           
            tnbr_inliers=sum(errors<=threshold);
            if tnbr_inliers>P3_inliers
                P3_inliers=tnbr_inliers;
                P3=tP3;
            end
        end
    end
end
P=[P,P3];
% Mark all useful information and display it
center=-inv(P3(:,1:3))*P3(:,4);
messages(num1).round=num1;
messages(num1).nbr_Us=nU;
messages(num1).inliers=P3_inliers;
messages(num1).inlier_rate=P3_inliers/sum(l);
messages(num1).center=center;
disp(['round: ',num2str(num1),', nU: ',num2str(nU),', inliers: ',num2str(P3_inliers),', inlier_rate: ',num2str(P3_inliers/sum(l)),', center: ',num2str(center')]);

%Update U for all feature groups with new features
U=[];
for i=1:length(track)
    if track(i).image_ids(end)==num2 
        l=length(track(i).image_ids);
        us=track(i).feature;
        Ps=P(end-l+1:end);      
        Uhat = ransac_triangulation(Ps, us, threshold);
        errors=reprojection_errors(Ps, us, Uhat);
        % Only update good U 
        index=find(errors<=threshold);
        if length(index)>1
            track(i).triangulated_point = refine_triangulation(Ps(index), us(:,index), Uhat);
        end      
    end
    U(:,i)=track(i).triangulated_point;
end
% Remove bad feature tracks
removed_indices=clean_outliers(U);
track=track(~removed_indices);

