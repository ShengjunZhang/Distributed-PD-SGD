function [W1, W2] = UpdateW(X, D, W1, W2, eta)

    X=X';D=D';   % Inputs

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
    W2 = W2 - eta*delta_i*(Z1.');
    W1 = W1 - eta*delta_j*(X.');

end
    
