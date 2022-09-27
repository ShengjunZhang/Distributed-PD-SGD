function [dW1, dW2] = GetSG(X, D, W1, W2)

    X=X';D=D';

    % activation function
    f = @(x) (1./(1+exp(-x)));

    % Propagate the signals through network
    a1 = W1*X;
    Z1 = [1;f(a1)];
    a2 = W2*Z1;
    Y = f(a2);

    % Output layer error
    delta_i = (Y-D);

    % Calculate error for each node in layer_(n-1)
    W20 = W2(:,2:end);
    delta_j = f(a1).*(1-f(a1)).*(W20.'*delta_i);

    % Adjust weights in matrices sequentially
    dW2 = delta_i*(Z1.');
    dW1 = delta_j*(X.');
    
end
    