function RMS_Err = GetError(Attributes, Classifications, W1, W2)


    % activation function
    f = @(x) (1./(1+exp(-x)));

    A1 = f(Attributes * W1');
    A1 = [ones(size(A1,1), 1) A1];
    A2 = A1 * W2';
    Y = f(A2);
    RMS_Err = sum(sum(Classifications.*log(Y) + (1-Classifications).*log(1-Y), 2));

end
