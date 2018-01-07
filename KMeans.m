function [quantized_mat,distorion,QL] = Quantizer(data,levels,meps)
%% Parameters Initialization
max_iter=20;
[m,s,n] = size(data);
data_flat = double(reshape(data,m,n*s));

%% Choose Random Initial levels
unique_levels = unique(data_flat,'rows');
QL            = unique_levels ( randperm ( size ( unique_levels,1 ),levels ),:);

%% Algorithm's Core
i=1;distorion=zeros(1,2);
%loop until difference is small, or reached max. iterations
while(i<max_iter && (i==1 || abs(distorion(i)-distorion(i-1)) > meps))
    %reshape centroids to a 3D matrix 
    QL_mat = ones(m*n,s*levels) .* reshape(QL',1,[]);
    QL_mat = reshape(QL_mat,m*n,s,levels);
    
    %calculate distance from each point to each centroid
    distance = sum((repmat(data_flat,1,1,levels) - QL_mat).^2,2);
    [~,quantized_mat] = min(distance,[],3);
    dataout = QL(quantized_mat,:);
    i=i+1;
    distorion(i) = mean(mean((dataout-data_flat).^2));
    
    %create a 3D with #levels sheets. Each contains all pixels which to the
    %relevant sheet
    idx_mat = zeros(m*n,s+1,levels);
    idx_mat(:,1:s,:) = repmat(data_flat,1,1,levels);
    idx_mat(:,s+1,:) = (reshape(quantized_mat == 1:levels,m*n,1,levels));
    idx_mat(:,1:s,:) = idx_mat(:,1:s,:).*idx_mat(:,s+1,:);

    %new centroid is calculated as the mean of all current pixels
    total= permute(sum(idx_mat),[3,2,1]);
    QL = total(:,1:s)./(total(:,s+1)+1);
end

dataout = reshape(dataout,m,n,s);