function [Err, Err_indx, cnt, MSE, W1, W2, time] = c_SGD(InputData, InputLabel, W1, W2,...
                            Attributes0, Classifications0, nbrOfEpochs, alpha0)
                        
    fprintf('Starting Centralized SGD.\n');
    tic;
    m = 0;                      % counter for iterations
    e = size(InputData);        % size of training data

    m_max = nbrOfEpochs*e(1); % max number of iterations

    % deciding when to do cost evaluation
    cnt = [1:9];
    for k=2:1:log10(m_max)
        cnt0 = [10^(k-1):5*10^(k-2):10^k - 1];
        cnt = [cnt cnt0];
    end
    cnt = [cnt 10^(log10(m_max))];

    % initialization
    k = 1;
    MSE = zeros(1,length(cnt));
    upd = textprogressbar(m_max);
    for m = 1:m_max

        g0 = max(1,log(1e-5*m));
        if m>1e5
            g0=g0*m/1e4;
        end
        gamma = 1e-5*g0;
        alpha = alpha0/(1+gamma*m);  

        % Iterate through one sample at a time
        ind0 = randi([1 e(1)],1,1);  % random index
        [W1, W2] = UpdateW(InputData(ind0,:), InputLabel(ind0,:), W1, W2, alpha);

        % compute the cost at prescribed iterations
        if m==cnt(k)
            RMS_Err = GetError(InputData, InputLabel, W1, W2);     % classification error
            MSE(k) = -RMS_Err + 0*norm(W2) + 0*norm(W1);            % Loss or cost we are minimizing
            k=k+1;
        end
        upd(m);
    end



    %% Testing
    
    e1 = -e(1)+length(Attributes0);
    if e1==0
        Attributes = [ones(e(1),1) Attributes0(1:e(1),:)];
        Classifications = Classifications0;
    else
        Attributes = [ones(e1(1),1) Attributes0(e(1)+1:end,:)];
        Classifications = Classifications0(e(1)+1:end,:);
    end

    [Err, Err_indx] = Testing(Attributes, Classifications, W1, W2);
    time = toc;
    fprintf('Centralized SGD done, and the running time is %d s.\n', time);
    
end