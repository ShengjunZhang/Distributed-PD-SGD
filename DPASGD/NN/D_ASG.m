function [Err, Err_indx, cnt, MSE, W1, W2, time] = D_ASG(InputData, InputLabel, W1, W2,...
         Attributes0, Classifications0, nbrOfEpochs, beta0, alpha0, e0, n, W)
     
    fprintf('Starting D_ASG.\n');
    tic;
    m_max = nbrOfEpochs*e0(1); % max number of iterations
    % initializarion of stochastic gradients
    
    gW1 = cell(1, n);
    gW2 = cell(1, n);
    
    
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
    
    % We need to initalize a Y before Loop with the same size of W
    % respectively, according to the paper, W @ time 0 and W @ time -1
    % could be chosen randomly, so, for similicity, let W @ time -1 be zero
    % and then Y1 and Y2 @ time could be (1+beta)*W1 and (1+beta)*W2. 
    
    alpha = alpha0;
    beta = beta0;
    
    
    Y1 = cell(1, n);
    Y2 = cell(1, n);
    
    for i = 1:n
        Y1{i} = (1+beta) * W1{i};
        Y2{i} = (1+beta) * W2{i};
    end
    
    
    for m = 1:m_max

        % Iterate through one sample at a time
        for i=1:n
            [mn, ~] = size(InputData{i});
            ind = randi([1 mn],1,1);
            [gW1{i}, gW2{i}] = GetSG(InputData{i}(ind,:), InputLabel{i}(ind,:), Y1{i}, Y2{i});
        end
        
        % Distributed stochastic update
        [W1, W2, Y1, Y2] = SGupdate_D_ASG(W1, W2, Y1, Y2, gW1, gW2, n, W, beta, alpha);

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
    fprintf('D_ASG done, and the running time is %d s.\n', time);
    
end