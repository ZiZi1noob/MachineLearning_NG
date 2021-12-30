function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% theta (28, 1)

X_parameter = theta' * X' ; %(1,100)
h_theta = sigmoid(X_parameter')'; %(1, 100);

part_a = log(h_theta) .* (-y)'; % 1,100
part_b = log(1 - h_theta) .* (1 - y)'; %(1,100)
J = sum(part_a - part_b, 2) / m; % 1 1

cost_part_b =sum(theta(2 : end) .^2) * lambda / (2*m);

J = J + cost_part_b;

grad = (((h_theta .- y') * X) ./ m)';
add = (theta .* lambda) ./ m;
grad(2 : size(theta)) =  grad(2 : size(theta)) .+ add(2 : size(theta));
%size(((h_theta .- y') * X) ./ m)  (1, 28)
%size((theta .* lambda) ./ m) (28, 1)



% =============================================================

end
