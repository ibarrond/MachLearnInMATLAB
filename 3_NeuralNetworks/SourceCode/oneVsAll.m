function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

    
    [m, n] = size(X);                       % Sizes
    all_theta = zeros(num_labels, n + 1);   % Initialized answer
    X = [ones(m, 1) X];                     % Bias neuron

    options = optimset('GradObj', 'on', 'MaxIter', 50);

    for c = 1:num_labels
        initial_theta = zeros(n + 1, 1);
        all_theta(c,:)=(fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
               initial_theta, options))';

    end
end
