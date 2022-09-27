%% This is used for plot results.

clear; clc; close all;



load('results.mat');
% load('results_c_SGD.mat');
% load('results_DSGD.mat');
% load('results_D_2.mat');
% load('results_m_SGD.mat');
% load('results_D_ASG.mat');

%%
% d = cnnConfig.layer{1}.dimension;
% testImages = loadMNISTImages('t10k-images-idx3-ubyte');
% testImages = reshape(testImages,d(1),d(2),d(3),[]);
% testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
% testLabels(testLabels==0) = 10; % Remap 0 to 10
% 
% %% cSGD
% [cost,grad,preds]=cnnCost(opttheta,testImages,testLabels,cnnConfig,meta,true);
% acc = sum(preds==testLabels)/length(preds);
% fprintf('c_SGD Accuracy is %f\n',acc);
% 
% %% DSGPA_F
% [cost_DSGPA_F,grad_DSGPA_F,preds_DSGPA_F]=cnnCost(opttheta_DSGPA_F,testImages,testLabels,cnnConfig,meta,true);
% acc_DSGPA_F = sum(preds_DSGPA_F==testLabels)/length(preds_DSGPA_F);
% fprintf('DSGPA_F accuracy is %f\n',acc_DSGPA_F);
% 
% %% DSGPA_T
% [cost_DSGPA_T,grad_DSGPA_T,preds_DSGPA_T]=cnnCost(opttheta_DSGPA_F,testImages,testLabels,cnnConfig,meta,true);
% acc_DSGPA_T = sum(preds_DSGPA_T==testLabels)/length(preds_DSGPA_T);
% fprintf('DSGPA_T accuracy is %f\n',acc_DSGPA_T);
% 
% %% DSGD_Jemin
% % [cost_DSGD_Jemin,grad_DSGD_Jemin,preds_DSGD_Jemin]=cnnCost(opttheta_DSGD_Jemin,testImages,testLabels,cnnConfig,meta,true);
% % acc_DSGD_Jemin = sum(preds_DSGD_Jemin==testLabels)/length(preds_DSGD_Jemin);
% % fprintf('DSGD_Jemin accuracy is %f\n',acc_DSGD_Jemin);
% 
% %% DSGD
% [cost_DSGD,grad_DSGD,preds_DSGD]=cnnCost(opttheta_DSGD,testImages,testLabels,cnnConfig,meta,true);
% acc_DSGD = sum(preds_DSGD==testLabels)/length(preds_DSGD);
% fprintf('DSGD accuracy is %f\n',acc_DSGD);
% 
% %% D^2
% % [cost_D_2,grad_D_2,preds_D_2]=cnnCost(opttheta_D_2,testImages,testLabels,cnnConfig,meta,true);
% % acc_D_2 = sum(preds_D_2==testLabels)/length(preds_D_2);
% % fprintf('D^2 accuracy is %f\n',acc_D_2);
% 
% %% Momentum SGD
% [cost_m_SGD,grad_m_SGD,preds_m_SGD]=cnnCost(opttheta_m_SGD,testImages,testLabels,cnnConfig,meta,true);
% acc_m_SGD = sum(preds_m_SGD==testLabels)/length(preds_m_SGD);
% fprintf('Momentum SGD accuracy is %f\n',acc_m_SGD);
% 
% %% D_ASG
% [cost_D_ASG,grad_D_ASG,preds_D_ASG]=cnnCost(opttheta_D_ASG,testImages,testLabels,cnnConfig,meta,true);
% acc_D_ASG = sum(preds_D_ASG==testLabels)/length(preds_D_ASG);
% fprintf('D-ASG accuracy is %f\n',acc_D_ASG);


l = 60000/n_agents;
l_c_SGD = 60000;
minibatch_c_SGD = 120;


%% Plot

figure;
set(gca,'FontSize', 15);

plot(1:options.epochs, sum(cost_e_DSGPA_F, 2)/(l/options.minibatch), '-o', 'MarkerSize',8,'LineWidth',2); hold on;
plot(1:options.epochs, sum(cost_e_DSGPA_T, 2)/(l/options.minibatch), '--s', 'MarkerSize',8,'LineWidth',2);
plot(1:options.epochs, sum(cost_e_m_SGD, 2)/(l/options.minibatch), '-.+', 'MarkerSize',8,'LineWidth',2);
plot(1:options.epochs, sum(cost_e_DSGD, 2)/(l/options.minibatch), '-x', 'MarkerSize',8,'LineWidth',2);
plot(1:options.epochs, sum(cost_e_D_ASG, 2)/(l/options.minibatch), '--^', 'MarkerSize',8,'LineWidth',2);
plot(1:options.epochs, sum(cost_e, 2)/(l_c_SGD/minibatch_c_SGD), 'k-', 'MarkerSize',8,'LineWidth',2); 

xticks(1:1:options.epochs);

xlabel('Epochs','Interpreter', 'latex', ...
        'FontSize', 15, 'FontWeight','bold');
ylabel('Training loss','Interpreter', 'latex', ...
        'FontSize', 15, 'FontWeight','bold');
    
legend({'DPA-SGD-F', ...
        'DPA-SGD-T', ...
        'DM-SGD in [15]', ...
        'D-SGD in [18] and [19]', ...
       'D-ASG in [36]', ...
       'C-SGD'}, ...
       'Interpreter', 'latex', ...
       'FontSize', 10, 'FontWeight','bold');    
    

   
   
   
%% Useless

% plot(1:options.epochs, sum(cost_e_D_2, 2)/(l/options.minibatch), '-', 'LineWidth', 2); 
% plot(1:options.epochs, sum(cost_e_DSGD_Jemin, 2)/(l/options.minibatch), '-.', 'LineWidth', 2);
% legend({'c\_SGD(Baseline)', ...
%         'DSGPA\_F', ...
%        'DSGPA\_T', ...
%        'DSGD\_Jemin', ...
%        'DSGD', ...
%        '$D^2$', ...
%        'Momentum SGD', ...
%        'D-ASG'}, ...
%        'Interpreter', 'latex', ...
%        'FontSize', 10, 'FontWeight','bold');
%    