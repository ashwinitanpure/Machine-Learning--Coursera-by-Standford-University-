function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
sum1=0;
sum2=0;


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %temp0 = theta(1,iter)-alpha*((theta(1)+X*(theta(2))-y)*X(iter));
    %temp1 = theta(2,iter)-alpha*((theta(1)+X*(theta(2))-y)*X(iter));
    %theta(1)=temp0;
   % theta(2)=temp1;
    x=X(:,2);
    hypothesis = theta(1) + (theta(2)*x);
    sum1 = sum(hypothesis - y);
    sum2 = sum((hypothesis - y) .*x);
   
    
    theta_1=theta(1)-(alpha/m)*sum1;
    theta_2=theta(2)-(alpha/m)*sum2;
    
    theta=[theta_1; theta_2];
   
    



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
   

end

end
