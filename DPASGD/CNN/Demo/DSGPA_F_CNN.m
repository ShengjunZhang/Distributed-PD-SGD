%% DSGPA with fixed parametersConvolution Neural Network

clear; clc; close all;

%% Initialize Parameters and Load Data for each agent

% Load MNIST Data from prepared data, the data has been distributed
% assgined into agents, in this case, we have 10 agents.

load('d_training.mat');

% Build CNN for agents, and each agent has the same network. But the
% weights are intilized randomly in cnnInitParams.m

cnn_distributed = cell(n_agents, 1);
cnnConfig = config();
[theta, meta] = cnnInitParams(cnnConfig);

for i = 1:n_agents
    cnn_distributed{i} = cnnInitParams(cnnConfig);
end

% Graph
type = 3;                       % 1 = ring, 2 = path, 3 = some random
[Lap, W_graph] = generateGraph(n_agents, type);


%% Training

options.epochs = 1;
options.minibatch = 60;
options.alpha = 1e-5;
options.eta = 1;
options.beta = 1;
options.momentum = 0.0;


% [opttheta, cost_e] = cnn_DSGPA_F(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta),theta,images,labels,options);
[opttheta, cost_e] = cnn_DSGPA_F(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta), cnn_distributed, training, Lap, n_agents, options);


%% Test
%  Test the performance of the trained model using the MNIST test set.

d = cnnConfig.layer{1}.dimension;
testImages = loadMNISTImages('t10k-images-idx3-ubyte');
testImages = reshape(testImages,d(1),d(2),d(3),[]);
testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10
opttheta = opttheta';
[cost,grad,preds]=cnnCost(opttheta,testImages,testLabels,cnnConfig,meta,true);

acc = sum(preds==testLabels)/length(preds);

fprintf('Accuracy is %f\n',acc);

figure;
plot(sum(cost_e,1));

