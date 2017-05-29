function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Nº of training examples
m = length(y); 

% Cost
J=1/m*(-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)))+lambda/(2*m)*(theta(2:end)'*theta(2:end));

% Gradient
grad = zeros(size(theta));
grad(1)=1/m*(sigmoid(X*theta)-y)'*X(:,1);
grad(2:end)=1/m*((sigmoid(X*theta)-y)'*X(:,2:end))'+lambda/m*theta(2:end);

end
