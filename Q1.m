%% Q1 - Initialization
clear all; close all;clc

levels = 1:10;
meps = 1;
data_struct = load('BreastCancerData.mat');
data = data_struct.X;
samples_ammount = size(data,2);
real_labels = data_struct.y;
%%
distorion = zeros(1,length(levels));
correct = zeros(1,length(levels));
total_correct=0;
bestK = 1;
for K=1:length(levels)
    [our_labels,dist,~] = Quantizer(data',levels(K),meps);
    distorion(K)        = dist(end);
    total_correct=0;
    for i=1:levels(K)
        positives = sum(real_labels(our_labels == i));
        allocated = sum((our_labels == i));
        total_correct = total_correct + max(positives, allocated-positives);
        our_final_labels(i) = positives > allocated /2;
    end
    correct(K) = total_correct / samples_ammount;
    if (correct(K) >= correct(bestK))
        bestK =K;
        best_labels = our_labels;
        best_final_labels = our_final_labels;
    end
end

figure;plot(levels,distorion);title('distortion according to amount of levels')
figure;plot(levels,abs(diff([0 distorion])));
figure;plot(levels,correct);

%% PCA
mapcaplot(data',best_labels);