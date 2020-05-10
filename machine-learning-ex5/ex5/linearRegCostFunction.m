function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

##J = (1/(2*m))*sum((X*theta-y).^2) + (lambda/(2*m))*sum(theta(2:end).^2);
J = (1/(2*m))*sum((X*theta-y).^2);

[theta_rows, theta_cols] = size(theta);
if theta_rows==1
  theta_dim = theta_cols;
else
  theta_dim = theta_rows;  
end

reg_factor = 0;

for idx = 1 : theta_dim 
  grad(idx,1) = (1/m)*sum(((theta'*X')-y').*X(:,idx)');
  if idx>1
    grad(idx,1) = grad(idx,1) + (lambda/m)*theta(idx,1);
    reg_factor = reg_factor + theta(idx,1)^2;
  endif
endfor

J = J + (lambda/(2*m))*reg_factor;


##grad(1,1) = (1/m)*sum(((theta'*X')-y').*X(:,1)');
##grad(2,1) = (1/m)*sum(((theta'*X')-y').*X(:,2)') + (lambda/m)*theta(2,1);

##grad_for_0 = 0;
##grad_for_1 = 0;
##J_for = 0;
##for idx=1:m
##  grad_for_0 = grad_for_0 + (X(idx,:)*theta-y(idx))*X(idx,1);
##  grad_for_1 = grad_for_1 + (X(idx,:)*theta-y(idx))*X(idx,2);
##  J_for = J_for + (X(idx,:)*theta-y(idx))^2;
##endfor
##
##J_for = (1/(2*m))*J_for + (lambda/(2*m))*sum(theta.^2);
##grad_for_0 = (1/m)*grad_for_0;
##grad_for_1 = (1/m)*grad_for_1 + (lambda/m)*theta(2,1);


% =========================================================================

grad = grad(:);

end
