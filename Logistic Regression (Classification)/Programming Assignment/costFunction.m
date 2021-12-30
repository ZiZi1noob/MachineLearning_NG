function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


X_parameter = theta' * X' ; %(1,100)
h_theta = sigmoid(X_parameter')'; %(1, 100);

part_a = log(h_theta) .* (-y)'; % 1,100

part_b = log(1 - h_theta) .* (1 - y)'; %(1,100)
temp = part_a - part_b;
J = sum(temp, 2) / m;
grad =   ((h_theta .- y') * X) ./ m;





% =============================================================

end
