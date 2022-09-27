function [Err, Err_indx, cnt, MSE, W1, W2, time] = D_2(InputData, InputLabel, W1, W2,...
         Attributes0, Classifications0, nbrOfEpochs, gamma0, e0, n, W)
     
    fprintf('Starting D^2.\n');
    tic;
    m_max = nbrOfEpochs*e0(1); % max number of iterations
    
    % initializarion of stochastic gradients
    
    gamma = gamma0;
    
    gW1 = cell(1, n);
    gW2 = cell(1, n);
    
    gW1_prev = cell(1, n);
    gW2_prev = cell(1, n);
    
    W1_prev = cell(1, n);
    W2_prev = cell(1, n);
    
    % deciding when to do cost evaluation
    cnt = [1:9];
    for k=2:1:log10(m_max)
        cnt0 = [10^(k-1):5*10^(k-2):10^k - 1];
        cnt = [cnt cnt0];
    end
    cnt = [cnt 10^(log10(m_max))];

    % initialization
    k = 1;
    MSE = zeros(length(cnt),n);
    upd = textprogressbar(m_max);
    
    % According to the algorithm, we first need to sample a data from local
    % worker and compute the gradient.
    
    for i = 1:n
        [mn, ~] = size(InputData{i});
        ind = randi([1 mn],1,1);
        [gW1_prev{i}, gW2_prev{i}] = GetSG(InputData{i}(ind,:), InputLabel{i}(ind,:), W1{i}, W2{i});
    end
    
    % X0 is the init value and we need to store a copy of it to reuse 
    % them in the loop
    
    W1_prev_prev = W1;
    W2_prev_prev = W2;
    
    % Get X1 in the following rule, we need to reshape
    
    for i = 1:n
        W1_prev{i} = W1_prev_prev{i} - gamma*gW1_prev{i};
        W2_prev{i} = W2_prev_prev{i} - gamma*gW2_prev{i};
    end
    
    W1_prev_mat = W1_prev{1};
    W2_prev_mat = W2_prev{1};

    [n1_prev_mat, m1_prev_mat] = size(W1_prev_mat);
    [n2_prev_mat, m2_prev_mat] = size(W2_prev_mat);

    W1_prev_vec = reshape(W1_prev_mat, 1, n1_prev_mat*m1_prev_mat);
    W2_prev_vec = reshape(W2_prev_mat, 1, n2_prev_mat*m2_prev_mat);

    for i = 2 : n
        W1_prev_vec(i,:) = reshape(W1_prev{i}, 1, n1_prev_mat*m1_prev_mat);
        W2_prev_vec(i,:) = reshape(W2_prev{i}, 1, n2_prev_mat*m2_prev_mat);
    end

    W1_prev_vec = W*W1_prev_vec;
    W2_prev_vec = W*W2_prev_vec;
    
    for i = 1 : n
        W1_prev{i} = reshape(W1_prev_vec(i,:), n1_prev_mat, m1_prev_mat);
        W2_prev{i} = reshape(W2_prev_vec(i,:), n2_prev_mat, m2_prev_mat);
    end
    
    W1 = W1_prev;
    W2 = W2_prev;
    
    temp_W1 = W1_prev_prev;
    temp_W2 = W2_prev_prev;
    
    W1_prev = temp_W1;
    W2_prev = temp_W2;
    
    for m = 1 : m_max

        % Iterate through one sample at a time
        for i = 1:n
            [mn, ~] = size(InputData{i});
            ind = randi([1 mn],1,1);
            [gW1{i}, gW2{i}] = GetSG(InputData{i}(ind,:), InputLabel{i}(ind,:), W1{i}, W2{i});
        end

        % Distributed stochastic update
        [W1, W2, W1_prev, W2_prev, gW1_prev, gW2_prev] = SGupdate_D_2(W1, W2, gW1, gW2, W1_prev, W2_prev, gW1_prev, gW2_prev, n, W, gamma);

        % compute the cost at prescribed iterations
        if m==cnt(k)
            for i=1:n
                RMS_Err = GetError(InputData{i}, InputLabel{i}, W1{i}, W2{i});
                MSE(k,i) = -RMS_Err + 0*norm(W2{i}) + 0*norm(W1{i});
            end
            k=k+1;
        end
        upd(m);

    end

    %% Testing
    
    e1 = -e0(1)+length(Attributes0);
    if e1==0
        Attributes = [ones(e0(1),1) Attributes0(1:e0(1),:)];
        Classifications = Classifications0;
    else
        Attributes = [ones(e1(1),1) Attributes0(e0(1)+1:end,:)];
        Classifications = Classifications0(e0(1)+1:end,:);
    end

    for i=1:n
        [Err(i), Err_indx{i}] = Testing(Attributes, Classifications, W1{i}, W2{i});
    end
    time = toc;
    fprintf('D^2 done, and the running time is %d s.\n', time);
    
end