function [W1, W2, W1_prev, W2_prev, gW1_prev, gW2_prev] = SGupdate_D_2(W1, W2, gW1, gW2, W1_prev, W2_prev, gW1_prev, gW2_prev, n, W, gamma)

    W1_temp = cell(1, n);
    W2_temp = cell(1, n);
    for i = 1 : n
        W1_temp{i} = 2*W1{i} - W1_prev{i} - gamma*gW1{i} + gamma*gW1_prev{i};
        W2_temp{i} = 2*W2{i} - W2_prev{i} - gamma*gW2{i} + gamma*gW2_prev{i};
    end
    
    W1_prev = W1;
    W2_prev = W2;
    gW1_prev = gW1;
    gW2_prev = gW2;

    
    % Reshape communicate and reshape back
    
    
    W1mat = W1_temp{1};
    W2mat = W2_temp{1};

    [n1, m1] = size(W1mat);
    [n2, m2] = size(W2mat);

    W1vec = reshape(W1mat, 1, n1*m1);
    W2vec = reshape(W2mat, 1, n2*m2);

    for i = 2 : n
        W1vec(i,:) = reshape(W1_temp{i}, 1, n1*m1);
        W2vec(i,:) = reshape(W2_temp{i}, 1, n2*m2);
    end

    W1vec = W*W1vec;
    W2vec = W*W2vec;

    for i = 1 : n
        W1{i} = reshape(W1vec(i,:), n1, m1);
        W2{i} = reshape(W2vec(i,:), n2, m2);
    end
    
  
end
