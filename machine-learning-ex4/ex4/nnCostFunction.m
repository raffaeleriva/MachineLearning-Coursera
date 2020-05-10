function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% ---------------------------------------------------
% Part 1: Feedforward propagation and cost function J
% ---------------------------------------------------

A1 = X;
A1 = [ones(m, 1) A1];
Zeta2 = A1*Theta1';
A2 = sigmoid(Zeta2);
A2 = [ones(m, 1) A2];
Zeta3 = A2*Theta2';
A3 = sigmoid(Zeta3);

K = size(Theta2,1); % this is the output layer size - need it to compute y(i) in Kx1 dimension

% cost function (J(Theta)) calculated looping on the m inputs
for idx = 1 : m
  y_vect = zeros(K,1);
  y_vect(y(idx)) = 1;
  
  J = J + sum((-y_vect'.*log(A3(idx,:)))-((1-y_vect').*log(1-(A3(idx,:)))));
endfor

#unregularized cost function
J = (1/m)*J; 

#regularization factor - note, exclude all elements with index j=0 (i.e. column with index 1 in Octave)
regularization = (lambda/(2*m))*((sum(sum(Theta1(:, 2:end).^2)))+(sum(sum(Theta2(:, 2:end).^2))));
#regularized cost function
J = J + regularization; 
  

% Additionally: this is the cost function (J(Theta)) calculated using a vectorized formula
m_=numel(y');
n_=max(y');
idx_=sub2ind([n_ m_],y',1:m_);
Y_vect=zeros(n_,m_);
Y_vect(idx_)=1; % Y_vect is an expansion in matrix form of the y vector, with dimension K (output layer size) x m (input samples)

J_vectorized = (1/m)*sum(sum((-Y_vect'.*log(A3))-((1-Y_vect').*log(1-(A3))))) + regularization;

% note that I checked on the correctness of this implementation vs the "for-looped" implementation, I am not outputting the value J_vectorized but it's equal to J...

% ---------------------------------------------------
%                      end Part 1
% ---------------------------------------------------


% ---------------------------------
% Part 2: backpropagation algorithm
% ---------------------------------
DELTA_2 = 0;
DELTA_1 = 0;
delta3 = zeros(K,m);
delta2 = zeros(hidden_layer_size,m);

for idx = 1 : m
  delta3(:,idx) = A3'(:,idx)-Y_vect(:,idx);
  delta2(:,idx) = Theta2(:,2:end)'*delta3(:,idx).*sigmoidGradient(Zeta2'(:,idx)); % note that since I compute the sigmoid gradient of z2 (and not of a2), then I need to exclude the first column of Theta2 when computing delta2 
  DELTA_2 = DELTA_2 + delta3(:,idx)*A2(idx,:);
  DELTA_1 = DELTA_1 + delta2(:,idx)*A1(idx,:);
endfor

Theta2_grad = (1/m)*DELTA_2;
Theta1_grad = (1/m)*DELTA_1;

% ---------------------------------
%           end Part 2
% ---------------------------------


% -----------------------------------
% Part 3: Regularization of gradients
% -----------------------------------
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) +(lambda/m)*Theta2(:, 2:end); % here for regularization we need not to add the regularization factor to the theta gradients with j=0 (i.e. column with index 1 in Octave)
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) +(lambda/m)*Theta1(:, 2:end); % same consideration as in line above also applies to the Theta1 gradients 
% -----------------------------------
%           end Part 3
% -----------------------------------
 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
