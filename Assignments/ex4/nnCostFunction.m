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

% Part 1: Feedforward Algorithm

% Adding a column of ones for bias to input
X = [ones(m, 1), X];

a1 = X;

z2 = a1 * Theta1';  % Calculate first layer output
a2 = sigmoid(z2);   % Applying sigmoid function to first layer output

% Adding a column of ones for bias to first layer output
a2 = [ones(size(a2, 1), 1), a2];

z3 = a2 * Theta2';  % Calculate second layer output
a3 = sigmoid(z3);   % Applying sigmoid function to second layer output

% Calculation of output values
hThetaX = a3;

% Part 2: Backpropagation Algorithm

% Setting test data to be a binary vector of 1s and 0s
binY = zeros(m, num_labels);

for i = 1 : m
    % Set the ith position of each test vector to be a 1
    binY(i, y(i)) = 1;
end

% Cost Function
J = (1 / m) * sum(sum((-binY .* log(hThetaX)) - ((1 - binY) .* log(1 - hThetaX))));

for s = 1 : m
    a1 = X(s,:)';
    
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    % binaryY = (ones(num_labels))';
    % binaryY(y(s)) = 1;
    binaryY = (1:num_labels)'==y(s);
    
    delta3 = a3 - binaryY;
    delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
    
    % Remove delta of bias
    delta2 = delta2(2:end);
    
    Theta1_grad = Theta1_grad + (delta2 * a1');
    Theta2_grad = Theta2_grad + (delta3 * a2');
end

Theta1_grad = (1 / m) * Theta1_grad;
Theta2_grad = (1 / m) * Theta2_grad;

% Part 3: Regularization

regularization = (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)));

J = J + regularization;

Theta1_grad = Theta1_grad + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad = Theta2_grad + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
