function [Err, Err_indx, cnt, MSE, W1, W2, time] = Jemin_SGD(InputData, InputLabel, W1, W2,...
         Attributes0, Classifications0, nbrOfEpochs, alpha0, beta0, e0, n, L)
     
    fprintf('Starting Jemin SGD.\n');
    tic;
    m_max = nbrOfEpochs*e0(1); % max number of iterations

    % initializarion of stochastic gradients
    
    gW1 = cell(1,n);
    gW2 = cell(1,n);

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
    for m = 1:m_max

        % Iterate through one sample at a time
        for i=1:n
            [mn, ~] = size(InputData{i});
            ind = randi([1 mn],1,1);
            [gW1{i}, gW2{i}] = GetSG(InputData{i}(ind,:), InputLabel{i}(ind,:), W1{i}, W2{i});
        end

        g0 = max(1,log(1e-5*m));
        if m>1e5
            g0=g0*m/1e4;
        end
        gamma = 1e-5*g0;
        alpha = alpha0/(1+gamma*m); 
        beta = beta0/((1+gamma*m)^(3/10));
        
        % Distributed stochastic update
        [W1, W2] = SGupdate(W1, W2, gW1, gW2, n, L, alpha, beta);

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
    fprintf('Jemin SGD done, and the running time is %d s.\n', time);
    
end