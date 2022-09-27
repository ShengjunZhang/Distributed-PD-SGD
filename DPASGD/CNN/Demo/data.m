%% show data points

clear; clc; close all;

%  complete the config.m to config the network structure;

cnnConfig = config();

%  calling cnnInitParams() to initialize parameters

[theta, meta] = cnnInitParams(cnnConfig);

% Load MNIST Data
images = loadMNISTImages('train-images-idx3-ubyte');

d = cnnConfig.layer{1}.dimension;
images = reshape(images,d(1),d(2),d(3),[]);
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

%% Demo data

% randomly pick up data points from the data set
% size_demo = 9;
% 
% figure;
% set(gcf,'color',[1 1 1]);
% 
% for i = 1: size_demo
%     idx = randi([1, size(labels, 1)], 1, 1);
%     image_demo = 255*images(:, :, :, idx);
%     label_demo = labels(idx);
%     label_demo(label_demo==10) = 0;
%     subplot(sqrt(size_demo), sqrt(size_demo), i);
%     imshow(image_demo);
%     title(['index =  ', num2str(idx)], 'Interpreter', 'latex');
%     xlabel(['label =  ', num2str(label_demo)], 'Interpreter', 'latex');
% end


%% Distribute data into n_agents set i.i.d

n_agents = 10;

training.image = cell(n_agents, 1);
training.label = cell(n_agents, 1);

m = length(labels);
data_size = m/n_agents;

rp = randperm(m);

for i = 1:n_agents
    
    training.image{i} = images(:,:,:,rp((i-1)*data_size+1:i*data_size));
    training.label{i} = labels(rp( (i-1)*data_size+1:i*data_size));
    
end

%% check dataset

% randomly pick up data points from the training set{i}

set_check = randi([1, n_agents], 1, 1);

images_check = training.image{set_check};
labels_check = training.label{set_check};

size_demo = 9;

figure;
set(gcf,'color',[1 1 1]);

for i = 1: size_demo
    idx = randi([1, size(labels_check, 1)], 1, 1);
    image_demo = 255*images_check(:, :, :, idx);
    label_demo = labels_check(idx);
    label_demo(label_demo==10) = 0;
    subplot(sqrt(size_demo), sqrt(size_demo), i);
    imshow(image_demo);
    title(['index =  ', num2str(idx)], 'Interpreter', 'latex');
    xlabel(['label =  ', num2str(label_demo)], 'Interpreter', 'latex');
end


