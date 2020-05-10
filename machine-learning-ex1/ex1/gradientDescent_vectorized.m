function [theta, J_history, theta_vect, J_history_vect] = gradientDescent_vectorized(X, y, theta, theta_vect, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
J_history_vect = zeros(num_iters, 1);

theta_vect_tmp = zeros(2,1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    sum_theta_0 = 0;
    sum_theta_1 = 0;
    for idx = 1:m
      sum_theta_0 = sum_theta_0 + (X(idx,:)*theta - y(idx));
      sum_theta_1 = sum_theta_1 + ((X(idx,:)*theta - y(idx))*X(idx,2));
    end
    
    theta(1,1) = theta(1,1) - (alpha/m)*sum_theta_0;
    theta(2,1) = theta(2,1) - (alpha/m)*sum_theta_1;
 
    
    theta_vect(1,1) = theta_vect(1,1) - (alpha/m)*sum(theta_vect_tmp'*X'-y');
    theta_vect(2,1) = theta_vect(2,1) - (alpha/m)*sum((theta_vect_tmp'*X'-y').*X(:,2)');
 
    theta_vect_tmp = theta_vect;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    % Save the cost J in every iteration    
    J_history_vect(iter) = computeCost(X, y, theta_vect);
    
end

end
