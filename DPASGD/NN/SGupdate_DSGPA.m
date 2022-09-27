function [W1, W2, V1, V2] = SGupdate_DSGPA(W1, W2, V1, V2, gW1, gW2, n, L, alpha, beta, eta)

    W1mat = W1{1};
    W2mat = W2{1};
    
    V1mat = V1{1};
    V2mat = V2{1};
    
    
    [n1, m1] = size(W1mat);
    [n2, m2] = size(W2mat);
    
    [nv1, mv1] = size(V1mat);
    [nv2, mv2] = size(V2mat);

    W1vec = reshape(W1mat, [1, n1*m1]);
    W2vec = reshape(W2mat, [1, n2*m2]);
    
    V1vec = reshape(V1mat, [1, nv1*mv1]);
    V2vec = reshape(V2mat, [1, nv2*mv2]);

    % ~~~~~~~~~~~~~~~~~~~~~~~~~ Update w ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for i = 2:n
        W1vec(i, :) = reshape(W1{i}, [1, n1*m1]);
        W2vec(i, :) = reshape(W2{i}, [1, n2*m2]);
        
        V1vec(i, :) = reshape(V1{i}, [1, nv1*mv1]);
        V2vec(i, :) = reshape(V2{i}, [1, nv2*mv2]);
    end

    W1vec = ( eye(n) - eta*alpha*L )*W1vec - eta*beta*V1vec;
    W2vec = ( eye(n) - eta*alpha*L )*W2vec - eta*beta*V2vec;

    for i = 1:n
        W1{i} = reshape(W1vec(i,:),[n1, m1]) - eta*gW1{i};
        W2{i} = reshape(W2vec(i,:),[n2, m2]) - eta*gW2{i};
    end
    % ~~~~~~~~~~~~~~~~~~~~~~~~~ Update v ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    W1_v_update_mat = W1{1};
    W2_v_update_mat = W2{1};
    
    [nWv1, mWv1] = size(W1_v_update_mat);
    [nWv2, mWv2] = size(W2_v_update_mat);
    
    W1_v_update = reshape(W1_v_update_mat, [1, nWv1*mWv1]);
    W2_v_update = reshape(W2_v_update_mat, [1, nWv2*mWv2]);
        
    for i = 1:n
        W1_v_update(i, :) = reshape(W1{i}, [1, nWv1*mWv1]);
        W2_v_update(i, :) = reshape(W2{i}, [1, nWv2*mWv2]);
    end
    W1_V = eta*beta*L*W1_v_update;
    W2_V = eta*beta*L*W2_v_update;
     
    for i = 1:n
        V1_temp = V1vec(i, :) + W1_V(i, :);
        V1{i} = reshape(V1_temp, [nv1, mv1]);
        V2_temp = V2vec(i, :) + W2_V(i, :);
        V2{i} = reshape(V2_temp, [nv2, mv2]);
    end
    
end
