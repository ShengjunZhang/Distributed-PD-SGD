%% This the main code.

clear; clc; close all;

load('opt_data.mat');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%% D-ASG in [36] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
    beta_D_ASG  = 0.8;
    alpha_D_ASG = 0.1;
    
    [Err_D_ASG, Err_indx_D_ASG, cnt_D_ASG, MSE_D_ASG, ...
         W1_D_ASG, W2_D_ASG, time_D_ASG] = D_ASG(InputData, InputLabel, W1, W2, ...
                                                 Attributes0, Classifications0, ...
                                                 nbrOfEpochs, beta_D_ASG, ...
                                                 alpha_D_ASG, e0, node, W_graph);
    figure;
    loglog(cnt_D_ASG, 1/e0(1)*sum(MSE_D_ASG, 2), 'LineWidth', 2);
    
    
%% %%%%%%%%%%%%%%%%%%%%%%%%% DSGT in [29], [37] %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note, these two algorithms are the same
        
    alpha_DSGT_29_37  = 0.01;
    
    [Err_DSGT_29_37, Err_indx_DSGT_29_37, cnt_DSGT_29_37, MSE_DSGT_29_37, ...
         W1_DSGT_29_37, W2_DSGT_29_37, time_DSGT_29_37] = DSGT_29_37(InputData, InputLabel, W1, W2, ...
                                                                     Attributes0, Classifications0, ...
                                                                     nbrOfEpochs, alpha_DSGT_29_37, ...
                                                                     e0, node, W_graph);
    figure;
    loglog(cnt_DSGT_29_37, 1/e0(1)*sum(MSE_DSGT_29_37, 2), 'LineWidth', 2);


%% %%%%%%%%%%%%%%%%%%%%%%%%% DSGT in [28], [38] %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note, these two algorithms are the same, [28] is undirected graph, [38]
% is directed graph, in this case, they are the same. [28] named the
% algorithm GNSD
        
    alpha_DSGT_28_38  = 0.01;
    
    [Err_DSGT_28_38, Err_indx_DSGT_28_38, cnt_DSGT_28_38, MSE_DSGT_28_38, ...
         W1_DSGT_28_38, W2_DSGT_28_38, time_DSGT_28_38] = DSGT_28_38(InputData, InputLabel, W1, W2, ...
                                                                     Attributes0, Classifications0, ...
                                                                     nbrOfEpochs, alpha_DSGT_28_38, ...
                                                                     e0, node, W_graph);
    figure;
    loglog(cnt_DSGT_28_38, 1/e0(1)*sum(MSE_DSGT_28_38, 2), 'LineWidth', 2);



%% %%%%%%%%%%%%%%%%%%%%%%% Momentum SGD in [15] %%%%%%%%%%%%%%%%%%%%%%%%%%%

        
    beta_m_SGD  = 0.8;
    gamma_m_SGD = 0.1;
    
    [Err_m_SGD, Err_indx_m_SGD, cnt_m_SGD, MSE_m_SGD, ...
         W1_m_SGD, W2_m_SGD, time_m_SGD] = Momentum_SGD(InputData, InputLabel, W1, W2, ...
                                                 Attributes0, Classifications0, ...
                                                 nbrOfEpochs, beta_m_SGD, ...
                                                 gamma_m_SGD, e0, node, W_graph);
    figure;
    loglog(cnt_m_SGD, 1/e0(1)*sum(MSE_m_SGD, 2), 'LineWidth', 2);
            save('Momentum_SGD.mat', 'e0', ...
         'Err_m_SGD', 'Err_indx_m_SGD', 'cnt_m_SGD', ...
         'MSE_m_SGD', 'W1_m_SGD', 'W2_m_SGD', 'time_m_SGD');



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%% D^2 in [27] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    gamma_D_2 = 1e-3;
    
    W1_D_2_init = cell(1, node);
    W2_D_2_init = cell(1, node);
    
    for i = 1:node
        W1_D_2_init{i} = zeros(size(W1{i}));
        W2_D_2_init{i} = zeros(size(W2{i}));
    end
    
    [Err_D_2, Err_indx_D_2, cnt_D_2, MSE_D_2, ...
         W1_D_2, W2_D_2, time_D_2] = D_2(InputData, InputLabel, W1_D_2_init, W2_D_2_init, ...
                                        Attributes0, Classifications0, ...
                                        nbrOfEpochs, gamma_D_2, e0, node, W_graph);
    figure;
    loglog(cnt_D_2, 1/e0(1)*sum(MSE_D_2, 2), 'LineWidth', 2);



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%% DSGPA_T %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    alpha_DSGPA_T   = 4;
    beta_DSGPA_T    = 3;
    eta_DSGPA_T     = 0.08;
    
    [Err_DSGPA_T, Err_indx_DSGPA_T, cnt_DSGPA_T, MSE_DSGPA_T, ...
         W1_DSGPA_T, W2_DSGPA_T, timeDSGPA_T] = DSGPA_T(InputData, InputLabel, W1, W2, V1, V2, ...
                                                        Attributes0, Classifications0, ...
                                                        nbrOfEpochs, alpha_DSGPA_T, ...
                                                        beta_DSGPA_T, eta_DSGPA_T, e0, node, Lap);
    figure;
    loglog(cnt_DSGPA_T, 1/e0(1)*sum(MSE_DSGPA_T, 2), 'LineWidth', 2);
            save('DSGPA_T.mat', 'e0', ...
         'Err_DSGPA_T', 'Err_indx_DSGPA_T', 'cnt_DSGPA_T', ...
         'MSE_DSGPA_T', 'W1_DSGPA_T', 'W2_DSGPA_T', 'time_DSGPA_T');



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%% DSGPA_F %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    alpha_DSGPA_F   = 5;
    beta_DSGPA_F    = 20;
    eta_DSGPA_F     = 0.03;
    
    [Err_DSGPA_F, Err_indx_DSGPA_F, cnt_DSGPA_F, MSE_DSGPA_F, ...
         W1_DSGPA_F, W2_DSGPA_F, timeDSGPA_F] = DSGPA_F(InputData, InputLabel, W1, W2, V1, V2, ...
                                                        Attributes0, Classifications0, ...
                                                        nbrOfEpochs, alpha_DSGPA_F, ...
                                                        beta_DSGPA_F, eta_DSGPA_F, e0, node, Lap);
                                                    
    loglog(cnt_DSGPA_F, 1/e0(1)*sum(MSE_DSGPA_F, 2), 'LineWidth', 2);
        save('DSGPA_F.mat', 'e0', ...
         'Err_DSGPA_F', 'Err_indx_DSGPA_F', 'cnt_DSGPA_F', ...
         'MSE_DSGPA_F', 'W1_DSGPA_F', 'W2_DSGPA_F', 'time_DSGPA_F');

%% %%%%%%%%%%%%%%%%%%%%%%% Centralized SGD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    alpha_c_SGD = 0.1;
    [Err_c_SGD, Err_indx_c_SGD, cnt_c_SGD, ...
     MSE_c_SGD, W1_c_SGD, W2_c_SGD, time_c_SGD] = c_SGD(Attributes, Classifications, ...
                                                        W10_avg, W20_avg, ...
                                                        Attributes0, ...
                                                        Classifications0, ...
                                                        nbrOfEpochs*1, alpha_c_SGD);
    save('c_SGD.mat', 'e0', ...
         'Err_c_SGD', 'Err_indx_c_SGD', 'cnt_c_SGD', ...
         'MSE_c_SGD', 'W1_c_SGD', 'W2_c_SGD', 'time_c_SGD');

%% %%%%%%%%%%%%%%%%%%% Distributed SGD in [18] [19] %%%%%%%%%%%%%%%%%%%%%%%

    alpha_DSGD_18_19 = 0.1;                   % initial value for alpha
    
    [Err_DSGD_18_19, Err_indx_DSGD_18_19, cnt_DSGD_18_19, MSE_DSGD_18_19, ...
     W1_DSGD_18_19, W2_DSGD_18_19, time_DSGD_18_19] = DSGD_18_19(InputData, InputLabel, W1, W2, ...
                                                             Attributes0, Classifications0, ...
                                                             nbrOfEpochs, alpha_DSGD_18_19, ...
                                                             e0, node, W_graph);
    figure;
    loglog(cnt_DSGD_18_19, 1/e0(1)*sum(MSE_DSGD_18_19, 2), 'LineWidth', 2);
                                                    
    
     
%% %%%%%%%%%%%%%%%%%%%%%%% Jemin SGD in [20] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    alpha_Jemin = 0.1;                   % initial value for alpha
    beta_Jemin = 1.01/max(svd(Lap));     % initial value for beta

    [Err_Jemin_SGD, Err_indx_Jemin_SGD, cnt_Jemin_SGD, MSE_Jemin_SGD, ...
     W1_Jemin_SGD, W2_Jemin_SGD, time_Jemin_SGD] = Jemin_SGD(InputData, InputLabel, W1, W2, ...
                                                             Attributes0, Classifications0, ...
                                                             nbrOfEpochs, alpha_Jemin, ...
                                                             beta_Jemin, e0, node, Lap);
                                                         
    save('Jemin.mat', 'e0', ...
         'Err_Jemin_SGD', 'Err_indx_Jemin_SGD', 'cnt_Jemin_SGD', ...
         'MSE_Jemin_SGD', 'W1_Jemin_SGD', 'W2_Jemin_SGD', 'time_Jemin_SGD');
     
%% Debug plot
% 
%     figure;
%     set(gca,'FontSize', 10);
%     loglog(cnt_c_SGD, 1/e0(1)*MSE_c_SGD);
%     loglog(cnt_Jemin_SGD, 1/e0(1)*sum(MSE_Jemin_SGD, 2)); 
%     xlabel('iteration $k$','Interpreter', 'latex', ...
%             'FontSize', 15, 'FontWeight','bold');
%     ylabel('Empirical Risk Function $R$','Interpreter', 'latex', ...
%             'FontSize', 15, 'FontWeight','bold');
