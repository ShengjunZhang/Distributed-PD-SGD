%% Centralized SGD Convolution Neural Network


% load('d_training.mat');

%% Initialize Parameters and Load Data

cnnConfig = config();

%  calling cnnInitParams() to initialize parameters

[theta, meta] = cnnInitParams(cnnConfig);

% Load MNIST Data
images = loadMNISTImages('train-images-idx3-ubyte');

d = cnnConfig.layer{1}.dimension;
images = reshape(images,d(1),d(2),d(3),[]);
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

%% Training

% images = training.image{1};
% labels = training.label{1};
l = length(labels);
options.epochs = 10;
options.minibatch = 200;
options.alpha = 1e-1;
options.momentum = 0.0;

[opttheta, cost_e] = minFuncSGD(@(x,y,z) cnnCost(x,y,z,cnnConfig,meta),theta,images,labels,options);


%% Test
%  Test the performance of the trained model using the MNIST test set.

testImages = loadMNISTImages('t10k-images-idx3-ubyte');
testImages = reshape(testImages,d(1),d(2),d(3),[]);
testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

%%
for i = 1:options.epochs
    test_opttheta = opttheta(i,:)';
    
    [cost,grad,preds]=cnnCost(test_opttheta,testImages,testLabels,cnnConfig,meta,true);
    
    acc(i) = sum(preds==testLabels)/length(preds);
    
    fprintf('Accuracy is %f\n',acc(i));
end
% figure;
% plot(1:options.epochs, sum(cost_e, 2)/(l/options.minibatch));

