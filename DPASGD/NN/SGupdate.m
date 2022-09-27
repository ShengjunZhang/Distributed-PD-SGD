function [W1, W2] = SGupdate(W1, W2, gW1, gW2, n, L, alpha, beta)

    W1mat = W1{1};
    W2mat = W2{1};

    [n1, m1] = size(W1mat);
    [n2, m2] = size(W2mat);

    W1vec = reshape(W1mat, 1, n1*m1);
    W2vec = reshape(W2mat, 1, n2*m2);

    % keyboard

    for i=2:n
        W1vec(i,:) = reshape(W1{i}, 1, n1*m1);
        W2vec(i,:) = reshape(W2{i}, 1, n2*m2);
    end

    W1vec = (eye(n)-beta*L)*W1vec;
    W2vec = (eye(n)-beta*L)*W2vec;

    for i=1:n
        W1{i} = reshape(W1vec(i,:),n1,m1) - alpha*gW1{i};
        W2{i} = reshape(W2vec(i,:),n2,m2) - alpha*gW2{i};
    end
    
end
