%% Machine Learning implemented in MATLAB - Notebook 3: Neural Networks
%
% _Author:_ Alberto Ibarrondo *|*
% _Date:_ 25/04/2017 *|*
% _License:_ MIT Free software
%


%% 1 INTRODUCTION
% The objective of this study is implement 1vsAll logistic regression, as 
% well as in neural networks for the MNIST dataset, which will allow us to 
% recognize hand-written digits.
% 
% *Files included with this notebook*
%
% * _MachLearnInMATLAB_3_NeuralNetworks.m_ - MATLAB script for the whole 
% implementation of the study. The script sets up the dataset for the 
% problems and makes calls to the rest of the functions. 
% * _data1.txt_ – Training set for the logistic regression
% * _data2.txt_ – Training set for the log. regression with regularization
% * _computeCost.m_ - Function to compute the cost of logistic regression 
% * _costFunctionReg.m_ - Regularized Logistic Regression Cost
% * _mapFeature.m_ - Function to generate polynomial features
% * _predict.m_ - Function to predict using logistic regression
%

%% 2 VECTORIZED LOGISTIC REGRESSION WITH REGULARIZATION

%% 2.1 Loading and Visualizing Data
% *Load Data*
%
% * X are 5000 examples with 20x20 pixels each, normalized and columnized
% * y is the label (1 to 10, being 10 the 0)
%

clear ; close all; clc
load('dataMNIST.mat'); % training data stored in arrays X, y
[m, n] = size(X); % number of training examples = 5000

% Set up Parameters
input_layer_size  = n;    % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10 (corresponding to 0)
lambda = 0.1;             % Regularization

fprintf('2.1 Loading and Visualizing Data ...\n')
fprintf('    Data Size: '); fprintf('%d ', [m, n]); fprintf('\n')
%%
% Visualize 36 random data elements
%
rand_indices = randperm(m);
sel = X(rand_indices(1:36), :);

displayMNIST(sel);

%% 2.2 Vectorized Logistic Regression
% In this part of the study we train several logistic regression models 
% with regularization, making sure that it can run vectorized. After
% that, one-vs-all classification is implemented for the handwritten
% digit dataset.
% 
fprintf('2.2 Vectorized One-vs-All Logistic Regression ...\n')
fprintf('    lambda = %.3f\n', lambda)

[all_theta] = oneVsAll(X, y, num_labels, lambda);

%% 2.2 Predict for One-Vs-All
% With the models trained before, we are gonna predict over the training
% set and calculate the accuracy
pred = predictOneVsAll(all_theta, X);
accuracy = mean(double(pred == y)) * 100;
fprintf('2.3 Predicting for One-Vs-All with log. regression models ...\n')
fprintf('    Train Accuracy: %f\n', accuracy);

