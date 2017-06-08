%% Machine Learning implemented in MATLAB - Notebook 6: KMeans & PCA
%
% _Author:_ Alberto Ibarrondo |
% _Date:_ 09/05/2017 |
% _License:_ MIT Free software
%

%% 1 INTRODUCTION
% The objective of this notebook is to implement some of the most common
% unsupervised learning algorithms: KMeans clustering to compress and image
% and Principal Component Analysis to find a low-dimensional representation
% of face images.
% 
% *Files included with this notebook*
%
% * _MachLearnInMATLAB_6_KMeansPCA.m_ - implementation in a MATLAB script
% * _data1.mat_ - Dataset for PCA
% * _data2.mat_ - Dataset for K-means
% * _faces.mat_ - Datasset for faces
% * _bird small.png - Example Image
% * _displayData.m_ - Displays 2D data stored in a matrix
% * _drawLine.m_ - Draws a line over an exsiting figure
% * _plotDataPoints.m_ - Plot a dataset
% * _plotProgresskMeans.m_ - Plots each step of K-means as it proceeds
% * _runkMeans.m_ - Runs the K-means algorithm
% * _pca.m_ - Perform principal component analysis
% * _projectData.m_ - Projects a data set into a lower dimensional space
% * _recoverData.m_ - Recovers the original data from the projection
% * _findClosestCentroids.m_ - Find closest centroids (used in K-means)
% * _computeCentroids.m_ - Compute centroid means (used in K-means)
% * _kMeansInitCentroids.m_ - Initialization for K-means centroids
%

clear ; close all; clc
addpath('data')
addpath('plotFunctions')

fprintf(2, '1. KMEANS & PCA\n')

%% 2 KMEANS CLUSTERING
% The intuition behind K-means is an iterative procedure that starts by 
% guessing the initial centroids, and then refines this guess by repeatedly
% assigning examples to their closest centroids and then recomputing the 
% centroids based on the assignments.

fprintf('2. KMEANS CLUSTERING\n')

%% 2.1 Find Closest Centroids 
% The first step in KMeans is to calculate the distance between each point
% and all the centroids in order to assess the closest centroid.

fprintf('2.1 Finding closest centroids.\n');
 
load('data2.mat');                      % Load the dataset
K = 3;                                  % Select a set of 3 centroids
initial_centroids = [3 3; 6 2; 8 5];

% Find the closest centroids for the examples using initial_centroids
idx = KMeans_ClosestCentroids(X, initial_centroids);

fprintf('     Closest centroids for the first 3 examples: \n')
fprintf('     %d', idx(1:3));
fprintf('\n     (the closest centroids should be 1, 3, 2)\n');


%% 2.2 Compute Means
%  After the centroids have been connected to the data, it's time to find
%  new centroids by averaging all the points assigned to a centroid.
%
fprintf('2.2 Computing centroids means.\n');

%  Compute means based on the closest centroids found in the previous part.
centroids = KMeans_ComputeCentroids(X, idx, K);

fprintf('    Updated Centroids: \n')
fprintf('    %f %f \n' , centroids');
fprintf('    (the centroids should be\n');
fprintf('  [ 2.428301 3.157924 ]\n');
fprintf('  [ 5.813503 2.633656 ]\n');
fprintf('  [ 7.119387 3.616684 ]\n\n');


%% 2.3 K-Means Clustering
%  After habing completed the two functions computeCentroids and
%  findClosestCentroids, we have all the necessary pieces to run the
%  kMeans algorithm. Let's run it!
%
fprintf('2.3 Running K-Means clustering on example dataset.\n\n');

load('data2.mat');                      % Load dataset
K = 3;  max_iters = 10;                 % Settings for running K-Means

initial_centroids =  KMeans_InitCentroids(X, K);  % Init. random centroids
KMeans(X, initial_centroids, max_iters, true);    % Run K-Means algorithm.


%% 2.4 KMeans Clustering on Pixels
%  After trying KMeans with a small dataset, we will use it to compress an 
%  image. To do this, we will first run it on the colors of the pixels and
%  then we will map each pixel on to it's closest centroid.
%  

fprintf('2.4 Running K-Means clustering on pixels from an image.\n');


A = double(imread('bird_small.png'));   %  Load an image of a bird
A = A / 255;                            % Normalize in range 0 - 1
img_size = size(A);                     % Size of the image

% Reshape the image into an Nx3 matrix where N = number of pixels.
X = reshape(A, img_size(1) * img_size(2), 3);

% Run K-Means algorithm on this data
K = 8; max_iters = 10;

% Random Initialization
initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

%% 2.5 Image Compression
% Finally, we will use the clusters of K-Means to compress an image. To do
% this, we first find the closest clusters for each example. 
fprintf('2.5 Applying K-Means to compress an image.\n\n');


idx = findClosestCentroids(X, centroids);   % Find closest cluster members

% Essentially, now we have represented the image X as in terms of the
% indices in idx. 
% We can now recover the image from the indices (idx) by mapping each pixel
% (specified by it's index in idx) to the centroid value
X_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display the original image 
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));




%% 3 PRINCIPAL COMPONENT ANALYSIS
fprintf('3 PRINCIPAL COMPONENT ANALYSIS - PCA\n');

%% 3.1 Loading & Plotting
fprintf('3.1 Visualizing example dataset for PCA.\n');

load ('data1.mat');                     % Variable X with 2 features

%  Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bo');  axis([0.5 6.5 2 8]); axis square;


%% 3.2 Normalizing Features
%  Before running PCA, it is important to first normalize X
fprintf('3.2 Normalizing the features.\n');

[X_norm, mu] = featureNormalize(X);                 % Normalize Features


%% 3.3 PCA Algorithm
[U, S] = pca(X_norm);
%  Compute mu, the mean of the each feature

%  Draw the eigenvectors centered at mean of data. These lines show the
%  directions of maximum variations in the dataset.
hold on;
drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
hold off;

fprintf('3.3 PCA Algorithm.\n');
fprintf('     Top eigenvector: \n');
fprintf('     U(:,1)  = %f %f \n', U(1,1), U(2,1));
fprintf('    (Expected: -0.707107 -0.707107)\n');


%% 3.4 Dimensionality Reduction
%  We implement now the projection to map the data onto the first k 
%  eigenvectors. 
%
fprintf('3.4 Dimension reduction\n');


plot(X_norm(:, 1), X_norm(:, 2), 'bo');     %  Plot the normalized dataset
axis([-4 3 -4 3]); axis square

%  Project the data onto K = 1 dimension
K = 1;
Z = projectData(X_norm, U, K);
fprintf('     Projection of the first example: %f\n', Z(1));
fprintf('     (this value should be about 1.481274)\n\n');

X_rec  = recoverData(Z, U, K);
fprintf('     Approximation of the first example: %f %f\n', X_rec(1, 1), X_rec(1, 2));
fprintf('     (this value should be about  -1.047419 -1.047419)\n\n');

%  Draw lines connecting the projected points to the original points
hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end
hold off

%% 3.4 Loading and Visualizing Face Data

fprintf('\nLoading face dataset.\n\n');

%  Load Face dataset
load ('ex7faces.mat')

%  Display the first 100 faces in the dataset
displayData(X(1:100, :));

%% =========== Part 5: PCA on Face Data: Eigenfaces  ===================
%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  We display the first 36 eigenfaces.
%
fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this mght take a minute or two ...)\n\n']);

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = pca(X_norm);

%  Visualize the top 36 eigenvectors found
displayData(U(:, 1:36)');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============= Part 6: Dimension Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors 
%  If you are applying a machine learning algorithm 
fprintf('\nDimension reduction for face dataset.\n\n');

K = 100;
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));


%% ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
%  Project images to the eigen space using the top K eigen vectors and 
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

K = 100;
X_rec  = recoverData(Z, U, K);


subplot(1, 2, 1);       % Display normalized data
displayData(X_norm(1:100,:));
title('Original faces');    axis square;



subplot(1, 2, 2);       % Display reconstructed data from only k eigenfaces
displayData(X_rec(1:100,:));
title('Recovered faces');   axis square;



%% 4. PCA & KMEANS FOR VISUALIZATION
%  One useful application of PCA is to use it to visualize high-dimensional
%  data. In the K-Means section we ran K-Means on 3-dimensional pixel 
%  colors of an image. We first visualize this output in 3D, and then
%  apply PCA to obtain a visualization in 2D.

%  Sample 1000 random indexes (since working with all the data is
%  too expensive. If you have a fast computer, you may increase this.
sel = floor(rand(1000, 1) * size(X, 1)) + 1;

%  Setup Color Palette
palette = hsv(K);
colors = palette(idx(sel), :);

%  Visualize the data and centroid memberships in 3D
figure;
scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
title('Pixel dataset plotted in 3D. Color shows centroid memberships');

%% === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
% Use PCA to project this cloud to 2D for visualization

% Subtract the mean to use PCA
[X_norm, mu, sigma] = featureNormalize(X);

% PCA and project the data to 2D
[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);

% Plot in 2D
figure;
plotDataPoints(Z(sel, :), idx(sel), K);
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
fprintf('Program paused. Press enter to continue.\n');
pause;

