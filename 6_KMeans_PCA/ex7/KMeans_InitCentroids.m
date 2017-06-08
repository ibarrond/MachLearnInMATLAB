function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X to be random examples
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X, using random examples
%
    randidx = randperm(size(X, 1));  % Randomly reorder the indices of examples
    centroids = X(randidx(1:K), :);  % Take the first K examples as centroids
end

