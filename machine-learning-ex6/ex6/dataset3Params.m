function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

len_C_vect = length(C_vect);
len_sigma_vect = length(sigma_vect);

err = zeros(len_C_vect, len_sigma_vect);

for idxC = 1 : len_C_vect
  for idxSigma = 1 : len_sigma_vect
    
    model= svmTrain(X, y, C_vect(idxC), @(x1, x2) gaussianKernel(x1, x2, sigma_vect(idxSigma)));
    predictions = svmPredict(model, Xval);

    err(idxC, idxSigma) = mean(double(predictions ~= yval));
    
  endfor
endfor


err_vect = err(:);

[min_err_vect, min_err_vect_idx] = min(err_vect);

col_min = ceil(min_err_vect_idx/len_C_vect);
row_min = rem(min_err_vect_idx,len_C_vect);

C = C_vect(row_min);
sigma = sigma_vect(col_min);


% =========================================================================

end
