function [J grad] = lrCostFunction(theta, X, y, lambda)

m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta);
J = (1/m) * (-y'*log(h) - (1-y)'*log(1-h)) + ((lambda/(2*m)) * sum(theta(2:length(theta)).^2));

grad_0 = (1/m) * (X'*(h - y));
theta(1) = 0;
grad_1_to_m= theta.*(lambda/m);
grad = grad_0 + grad_1_to_m;

grad = grad(:);

end