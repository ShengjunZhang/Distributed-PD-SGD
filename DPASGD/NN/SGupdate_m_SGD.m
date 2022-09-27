function [W1, W2, U1, U2] = SGupdate_m_SGD(W1, W2, U1, U2, gW1, gW2, n, W, beta, gamma)%rather than gamma, beta 
   
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Update ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i = 1:n
        U1{i} = beta*U1{i} + gW1{i};
        U2{i} = beta*U2{i} + gW2{i};

        W1{i} = W1{i} - gamma*U1{i};
        W2{i} = W2{i} - gamma*U2{i};
    end
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Communicate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    U1mat = U1{1};
    U2mat = U2{1};

    [nU1, mU1] = size(U1mat);
    [nU2, mU2] = size(U2mat);

    U1vec = reshape(U1mat, 1, nU1*mU1);
    U2vec = reshape(U2mat, 1, nU2*mU2);


    for i=2:n
        U1vec(i,:) = reshape(U1{i}, 1, nU1*mU1);
        U2vec(i,:) = reshape(U2{i}, 1, nU2*mU2);
    end

    U1vec = W*U1vec;
    U2vec = W*U2vec;

    for i=1:n
        U1{i} = reshape(U1vec(i,:),nU1,mU1);
        U2{i} = reshape(U2vec(i,:),nU2,mU2);
    end
    
   
    W1mat = W1{1};
    W2mat = W2{1};

    [n1, m1] = size(W1mat);
    [n2, m2] = size(W2mat);

    W1vec = reshape(W1mat, 1, n1*m1);
    W2vec = reshape(W2mat, 1, n2*m2);


    for i=2:n
        W1vec(i,:) = reshape(W1{i}, 1, n1*m1);
        W2vec(i,:) = reshape(W2{i}, 1, n2*m2);
    end

    W1vec = W*W1vec;
    W2vec = W*W2vec;

    for i=1:n
        W1{i} = reshape(W1vec(i,:),n1,m1);
        W2{i} = reshape(W2vec(i,:),n2,m2);
    end
    
end
