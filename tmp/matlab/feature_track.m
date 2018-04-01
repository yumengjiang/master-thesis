function track=feature_track(track,track_new)
k=length(track);
for i=1:length(track_new)
    flag=false;
    for j=1:length(track)
        index=find(track(j).image_ids==track_new(i).image_ids(1));
        if length(index)>1
            index
        end
        if length(index)>0
            if track(j).feature_ids(index)==track_new(i).feature_ids(1)            
                flag=true;
                track(j).image_ids=union(track(j).image_ids,track_new(i).image_ids(2),'stable');
                track(j).feature_ids=union(track(j).feature_ids,track_new(i).feature_ids(2),'stable');
%                 track(j).feature=union(track(j).feature',track_new(i).feature(:,2)','rows','stable')';
                track(j).triangulated_point=union(track(j).triangulated_point',track_new(i).triangulated_point','rows','stable')';
%                 track(j).image_ids=[track(j).image_ids,track_new(i).image_ids(2)];
%                 track(j).feature_ids=[track(j).feature_ids,track_new(i).feature_ids(2)];
%                 track(j).feature=[track(j).feature,track_new(i).feature(:,2)];
%                 track(j).triangulated_point=[track(j).triangulated_point,track_new(i).triangulated_point];
            end
        end
    end
    if flag==false
        k=k+1;
        track(k)=track_new(i);
    end
end