function [W1, W2, W1_prev, W2_prev, gW1_prev, gW2_prev] = SGupdate_DSGT_29_37(W1, W2, gW1, gW2, W1_prev, W2_prev, gW1_prev, gW2_prev, n, W, alpha)
    
    % Save a temp for W1, and W2
    
    W1_copy = W1;
    W2_copy = W2;

    % Reshape and communicate.
    
    W1_temp = W1;
    W2_temp = W2;
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
    

    W1_prev_temp = W1_prev;
    W2_prev_temp = W2_prev;
    
    W1_prev_mat = W1_prev_temp{1};
    W2_prev_mat = W2_prev_temp{1};

    [nW1_prev, mW1_prev] = size(W1_prev_mat);
    [nW2_prev, mW2_prev] = size(W2_prev_mat);

    W1_prev_vec = reshape(W1_prev_mat, 1, nW1_prev*mW1_prev);
    W2_prev_vec = reshape(W2_prev_mat, 1, nW2_prev*mW2_prev);

    for i = 2 : n
        W1_prev_vec(i,:) = reshape(W1_prev_temp{i}, 1, nW1_prev*mW1_prev);
        W2_prev_vec(i,:) = reshape(W2_prev_temp{i}, 1, nW2_prev*mW2_prev);
    end

    W1_prev_vec = W*W*W1_prev_vec;
    W2_prev_vec = W*W*W2_prev_vec;

    for i = 1 : n
        W1_prev{i} = reshape(W1_prev_vec(i,:), nW1_prev, mW1_prev);
        W2_prev{i} = reshape(W2_prev_vec(i,:), nW2_prev, mW2_prev);
    end
    
    
    
    gW1_diff = cell(1, n);
    gW2_diff = cell(1, n);
    
    for i = 1 : n
        gW1_diff{i} = gW1{i} - gW1_prev{i};
        gW2_diff{i} = gW2{i} - gW2_prev{i};
    end
    
    gW1_diff_mat = gW1_diff{1};
    gW2_diff_mat = gW2_diff{1};

    [ngW1_diff, mgW1_diff] = size(gW1_diff_mat);
    [ngW2_diff, mgW2_diff] = size(gW2_diff_mat);

    gW1_diff_vec = reshape(gW1_diff_mat, 1, ngW1_diff*mgW1_diff);
    gW2_diff_vec = reshape(gW2_diff_mat, 1, ngW2_diff*mgW2_diff);

    for i = 2 : n
        gW1_diff_vec(i,:) = reshape(gW1_diff{i}, 1, ngW1_diff*mgW1_diff);
        gW2_diff_vec(i,:) = reshape(gW2_diff{i}, 1, ngW2_diff*mgW2_diff);
    end

    gW1_diff_vec = W*gW1_diff_vec;
    gW2_diff_vec = W*gW2_diff_vec;

    for i = 1 : n
        gW1_diff{i} = reshape(gW1_diff_vec(i,:), [ngW1_diff, mgW1_diff]);
        gW2_diff{i} = reshape(gW2_diff_vec(i,:), [ngW2_diff, mgW2_diff]);
    end
    
       

    % Update.
    
    for i = 1 : n
        W1{i} = 2*W1{i} - W1_prev{i} - alpha*gW1_diff{i};
        W2{i} = 2*W2{i} - W2_prev{i} - alpha*gW2_diff{i};
    end
    
    W1_prev = W1_copy;
    W2_prev = W2_copy;
    gW1_prev = gW1;
    gW2_prev = gW2;

  
end
