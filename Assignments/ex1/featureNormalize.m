function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% Solution 1
% numFeatures = size(X, 2);
% 
% for i = 1 : numFeatures
%     average = mean(X(:, i));
%     mu(i) = average;
%     standardDeviation = std(X(:, i));
%     sigma(i) = standardDeviation;
%     X_norm(:, i) = (X_norm(:, i) - average) / standardDeviation;
% end

% Solution 2
mu = mean(X);
sigma = std(X);
X_norm = (X_norm - mu) ./ sigma;

% ============================================================

end
