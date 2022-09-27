%% DSGPA with fixed parametersConvolution Neural Network

clear; clc; 
close all;

%% Initialize Parameters and Load Data for each agent

% Load MNIST Data from prepared data, the data has been distributed
% assgined into agents, in this case, we have 10 agents.

load('d_training.mat');

% Build CNN for agents, and each agent has the same network. But the
% weights are intialized randomly in cnnInitParams.m for each agent.

cnn_distributed = cell(n_agents, 1);
cnnConfig = config();
[theta, meta] = cnnInitParams(cnnConfig);

for i = 1:n_agents
    cnn_distributed{i} = cnnInitParams(cnnConfig);
end

% Graph
type = 3;                       % 1 = ring, 2 = path, 3 = some random
[Lap, W_graph] = generateGraph(n_agents, type);

%%=========================================================================
%% Training Sessions

% % We compare different algorithms under the same graph, same data set and
% CNN model.

%% DSGPA_F

options.epochs = 10;
options.minibatch = 20;
options.alpha = 5e-1;
options.eta = 5e-1;
options.beta = 1e-1;


alpha = options.alpha;
eta = options.eta;

L = 1/(alpha*eta) * (eye(n_agents) - W_graph);

[opttheta_DSGPA_F, cost_e_DSGPA_F, time_DSGPA_F] = cnn_DSGPA_F(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta), cnn_distributed, training, L, n_agents, options);

%%=========================================================================
%% DSGPA_T

options.epochs = 10;
options.minibatch = 20;
options.alpha = 5e-1;
options.eta = 5e-1;
options.beta = 1e-1;

alpha = options.alpha;
eta = options.eta;

L = 1/(alpha*eta) * (eye(n_agents) - W_graph);

[opttheta_DSGPA_T, cost_e_DSGPA_T, time_DSGPA_T] = cnn_DSGPA_T(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta), cnn_distributed, training, L, n_agents, options);

%%=========================================================================
%% Jemin DSGD

% options.epochs = 10;
% options.minibatch = 20;
% 
% options.alpha = 5e-3;
% options.beta = 1.01/max(svd(Lap));
% 
% [opttheta_DSGD_Jemin, cost_e_DSGD_Jemin, time_DSGD_Jemin] = cnn_DSGD_Jemin(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta), cnn_distributed, training, W_graph, n_agents, options);

%%=========================================================================
%% DSGD
% % The following papers are both from NeurIPS 2017
% 1. Collaborative Deep Learning in Fixed Topology Networks.
% 2. Can Decentralized Algorithms Outperform Centralized Algorithms? A Case
% Study for Decentralized Parallel Stochastic Gradient Descent.

options.epochs = 10;
options.minibatch = 20;
options.gamma = 1e-1;

[opttheta_DSGD, cost_e_DSGD, time_DSGD] = cnn_DSGD(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta), cnn_distributed, training, W_graph, n_agents, options);

%%=========================================================================
%% D^2

% options.epochs = 10;
% options.minibatch = 20;
% options.gamma = 1e-1;
% 
% [opttheta_D_2, cost_e_D_2, time_D_2] = cnn_D_2(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta), cnn_distributed, training, W_graph, n_agents, options);

%%=========================================================================

%% Momentum SGD

options.epochs = 10;
options.minibatch = 20;
options.gamma = 1e-1;
options.beta = 8e-1;

[opttheta_m_SGD, cost_e_m_SGD, time_m_SGD] = cnn_m_SGD(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta), cnn_distributed, training, W_graph, n_agents, options);

%%=========================================================================

%% D-ASG

options.epochs = 10;
options.minibatch = 20;
options.alpha = 1e-1;
options.beta = 8e-1;

[opttheta_D_ASG, cost_e_D_ASG, time_D_ASG] = cnn_D_ASG(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta), cnn_distributed, training, W_graph, n_agents, options);

%%=========================================================================

%% DSGT in [28][38]

% options.epochs = 10;
% options.minibatch = 20;
% options.alpha = 8e-1;
% 
% [opttheta_DSGT_28_38, cost_e_DSGT_28_38, time_DSGT_28_38] = cnn_DSGT_28_38(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta), cnn_distributed, training, W_graph, n_agents, options);

%%=========================================================================

%% DSGT in [29][37]

% options.epochs = 10;
% options.minibatch = 20;
% options.alpha = 8e-1;
% 
% [opttheta_DSGT_29_37, cost_e_DSGT_29_37, time_DSGT_29_37] = cnn_DSGT_29_37(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta), cnn_distributed, training, W_graph, n_agents, options);

%%=========================================================================


%% Test
%  Test the performance of the trained model using the MNIST test set.

d = cnnConfig.layer{1}.dimension;
testImages = loadMNISTImages('t10k-images-idx3-ubyte');
testImages = reshape(testImages,d(1),d(2),d(3),[]);
testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

%% DSGPA_F

for i = 1:options.epochs
    test_opttheta_DSGPA_F = opttheta_DSGPA_F(i,:)';
    [cost_DSGPA_F,grad_DSGPA_F,preds_DSGPA_F]=cnnCost(test_opttheta_DSGPA_F,testImages,testLabels,cnnConfig,meta,true);
    acc_DSGPA_F(i) = sum(preds_DSGPA_F==testLabels)/length(preds_DSGPA_F);
    fprintf('DSGPA_F accuracy is %f\n',acc_DSGPA_F(i));
end

% opttheta_DSGPA_F = opttheta_DSGPA_F';
% [cost_DSGPA_F,grad_DSGPA_F,preds_DSGPA_F]=cnnCost(opttheta_DSGPA_F,testImages,testLabels,cnnConfig,meta,true);
% acc_DSGPA_F = sum(preds_DSGPA_F==testLabels)/length(preds_DSGPA_F);
% fprintf('DSGPA_F accuracy is %f\n',acc_DSGPA_F);

%% DSGPA_T

for i = 1:options.epochs
    test_opttheta_DSGPA_T = opttheta_DSGPA_T(i,:)';
    [cost_DSGPA_T,grad_DSGPA_T,preds_DSGPA_T]=cnnCost(test_opttheta_DSGPA_T,testImages,testLabels,cnnConfig,meta,true);
    acc_DSGPA_T(i) = sum(preds_DSGPA_T==testLabels)/length(preds_DSGPA_T);
    fprintf('DSGPA_T accuracy is %f\n',acc_DSGPA_T(i));
end


% opttheta_DSGPA_T = opttheta_DSGPA_T';
% [cost_DSGPA_T,grad_DSGPA_T,preds_DSGPA_T]=cnnCost(opttheta_DSGPA_F,testImages,testLabels,cnnConfig,meta,true);
% acc_DSGPA_T = sum(preds_DSGPA_T==testLabels)/length(preds_DSGPA_T);
% fprintf('DSGPA_T accuracy is %f\n',acc_DSGPA_T);

%% DSGD_Jemin
% opttheta_DSGD_Jemin = opttheta_DSGD_Jemin';
% [cost_DSGD_Jemin,grad_DSGD_Jemin,preds_DSGD_Jemin]=cnnCost(opttheta_DSGD_Jemin,testImages,testLabels,cnnConfig,meta,true);
% acc_DSGD_Jemin = sum(preds_DSGD_Jemin==testLabels)/length(preds_DSGD_Jemin);
% fprintf('DSGD_Jemin accuracy is %f\n',acc_DSGD_Jemin);

%% DSGD
for i = 1:options.epochs
    test_opttheta_DSGD = opttheta_DSGD(i,:)';
    [cost_DSGD,grad_DSGD,preds_DSGD]=cnnCost(test_opttheta_DSGD,testImages,testLabels,cnnConfig,meta,true);
    acc_DSGD(i) = sum(preds_DSGD==testLabels)/length(preds_DSGD);
    fprintf('DSGD accuracy is %f\n',acc_DSGD(i));
end
%% D^2
% opttheta_D_2 = opttheta_D_2';
% [cost_D_2,grad_D_2,preds_D_2]=cnnCost(opttheta_D_2,testImages,testLabels,cnnConfig,meta,true);
% acc_D_2 = sum(preds_D_2==testLabels)/length(preds_D_2);
% fprintf('D^2 accuracy is %f\n',acc_D_2);
% 
% l = length(training.label{1});
% plot(1:1:options.epochs, sum(cost_e_D_2, 2)/(l/options.minibatch), '-', 'LineWidth', 2);

%% Momentum SGD
for i = 1: options.epochs
    test_opttheta_m_SGD = opttheta_m_SGD(i,:)';
    [cost_m_SGD,grad_m_SGD,preds_m_SGD]=cnnCost(test_opttheta_m_SGD,testImages,testLabels,cnnConfig,meta,true);
    acc_m_SGD(i) = sum(preds_m_SGD==testLabels)/length(preds_m_SGD);
    fprintf('Momentum SGD accuracy is %f\n',acc_m_SGD(i));
end

% l = length(training.label{1});
% plot(1:1:options.epochs, sum(cost_e_m_SGD, 2)/(l/options.minibatch), '-', 'LineWidth', 2);

%% D-ASG
for i = 1: options.epochs
    test_opttheta_D_ASG = opttheta_D_ASG(i,:)';
    [cost_D_ASG,grad_D_ASG,preds_D_ASG]=cnnCost(test_opttheta_D_ASG,testImages,testLabels,cnnConfig,meta,true);
    acc_D_ASG(i) = sum(preds_D_ASG==testLabels)/length(preds_D_ASG);
    fprintf('D-ASG accuracy is %f\n',acc_D_ASG(i));
end
% l = length(training.label{1});
% plot(1:1:options.epochs, sum(cost_e_D_ASG, 2)/(l/options.minibatch), '-', 'LineWidth', 2);

%% DSGT in [28][38]
% opttheta_DSGT_28_38 = opttheta_DSGT_28_38';
% [cost_DSGT_28_38,grad_DSGT_28_38,preds_DSGT_28_38]=cnnCost(opttheta_DSGT_28_38,testImages,testLabels,cnnConfig,meta,true);
% acc_DSGT_28_38 = sum(preds_DSGT_28_38==testLabels)/length(preds_DSGT_28_38);
% fprintf('DSGT in [28][38] accuracy is %f\n',acc_D_ASG);
% 
% l = length(training.label{1});
% plot(1:1:options.epochs, sum(cost_e_DSGT_28_38, 2)/(l/options.minibatch), '-', 'LineWidth', 2);

%% DSGT in [29][37]
% opttheta_DSGT_29_37 = opttheta_DSGT_29_37';
% [cost_DSGT_29_37,grad_DSGT_29_37,preds_DSGT_29_37]=cnnCost(opttheta_DSGT_29_37,testImages,testLabels,cnnConfig,meta,true);
% acc_DSGT_29_37 = sum(preds_DSGT_29_37==testLabels)/length(preds_DSGT_29_37);
% fprintf('DSGT in [29][37] accuracy is %f\n',acc_D_ASG);
% 
% l = length(training.label{1});
% plot(1:1:options.epochs, sum(cost_e_DSGT_29_37, 2)/(l/options.minibatch), '-', 'LineWidth', 2);


%%
% % l = length(training.label{1});
% % 
% % figure;
% % 
% % set(gca,'FontSize', 15);
% % 
% % plot(1:1:options.epochs, sum(cost_e_DSGPA_F, 2)/(l/options.minibatch), '-', 'LineWidth', 2); hold on;
% % plot(1:1:options.epochs, sum(cost_e_DSGPA_T, 2)/(l/options.minibatch), '--', 'LineWidth', 2);
% % plot(1:1:options.epochs, sum(cost_e_DSGD_Jemin, 2)/(l/options.minibatch), '-.', 'LineWidth', 2);
% % plot(1:1:options.epochs, sum(cost_e_DSGD, 2)/(l/options.minibatch), ':', 'LineWidth', 2);
% % 
% % xticks(0:1:options.epochs);
% % 
% % xlabel('epoch $k$','Interpreter', 'latex', ...
% %         'FontSize', 15, 'FontWeight','bold');
% % ylabel('Loss','Interpreter', 'latex', ...
% %         'FontSize', 15, 'FontWeight','bold');
% % legend({'DSGPA\_F', ...
% %        'DSGPA\_T', ...
% %        'DSGD\_Jemin'}, ...
% %        'Interpreter', 'latex', 'FontSize', 10, 'FontWeight','bold');
   
%%=========================================================================
