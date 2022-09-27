function [W1, W2, Y1, Y2] = SGupdate_D_ASG(W1, W2, Y1, Y2, gW1, gW2, n, W, beta, alpha)
    
    % Save a copy of current W1, and W2
    
    W1_copy = W1;
    W2_copy = W2;

    % Reshape Y and communicate and reshape back
    
    Y1mat = Y1{1};
    Y2mat = Y2{1};

    [nY1, mY1] = size(Y1mat);
    [nY2, mY2] = size(Y2mat);

    Y1vec = reshape(Y1mat, 1, nY1*mY1);
    Y2vec = reshape(Y2mat, 1, nY2*mY2);


    for i=2:n
        Y1vec(i,:) = reshape(Y1{i}, 1, nY1*mY1);
        Y2vec(i,:) = reshape(Y2{i}, 1, nY2*mY2);
    end

    Y1vec = W*Y1vec;
    Y2vec = W*Y2vec;

    for i=1:n
        Y1{i} = reshape(Y1vec(i,:),nY1,mY1);
        Y2{i} = reshape(Y2vec(i,:),nY2,mY2);
    end
    
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Update ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i = 1:n
        W1{i} = Y1{i} - alpha*gW1{i};
        W2{i} = Y2{i} - alpha*gW2{i};

        Y1{i} = (1+beta)*W1{i} - beta*W1_copy{i};
        Y2{i} = (1+beta)*W2{i} - beta*W2_copy{i};
    end

    
end
