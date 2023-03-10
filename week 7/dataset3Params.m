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
sigma = 0.1;
%c = 0;
%s = 0;
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

%range = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%r = 10000000;
%for i = 1:8
%  for j = 1:8
%    model= svmTrain(X, y, range(i, 1), @(x1, x2) gaussianKernel(x1, x2, range(j, 1)));
%    predictions = svmPredict(model, Xval);
%    error = mean(double(predictions ~= yval));
%    if (error < r)
%      r = error;
%      c = i;
%      s = j;
%    endif
%  endfor
%endfor
%
%C = range(c, 1)
%sigma = range(s, 1)

% =========================================================================
end
