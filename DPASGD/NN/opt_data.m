%% This is used for prepare data set.

clear; clc; close all;

%%%%%%%%%%%%%%%%%%% Parameters and Graph Initilization %%%%%%%%%%%%%%%%%%%%

node = 10;          % nodes number
type = 3;           % 1 = ring, 2 = path, 3 = some random
[Lap, W_graph] = generateGraph(node, type);

%%%%%%%%%%%%%%%%%%%%%%%%% Data Preparation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DataSize = 5e3;
[X, y] = GetData(DataSize, 'dataset.mat');

% One hot encoding Y
Y = zeros(length(X),10);

for i=1:length(X)    
    Y(i,y(i))=1; 
end

% Saving Data
Attributes0 = X; Classifications0 = Y;
clear X Y;

e0 = round(size(Attributes0)*0.5); % data used for trining (using only half)

Attributes = [ones(e0(1),1) Attributes0(1:e0(1),:)];
Classifications = Classifications0(1:e0(1),:);



% Data for distributed training, divided data equally into 10 sets.

InputData = cell(1, node);
InputLabel = cell(1, node);


count = zeros(1, node);

for i =1: node
    InputData{i} = Attributes( (i-1)*e0(1)/node + 1 : i*e0(1)/node, :);
    InputLabel{i} = Classifications( (i-1)*e0(1)/node + 1 : i*e0(1)/node, :);
    count(i) = count(i) + length(InputLabel{i});
end

if count == e0(1)/node
    fprintf('Data has been eqaully distributed.\n');
end


%%%%%%%%%%%%%%%%%%%%%%%%% Neural Nets Design %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nbrOfNodes = 50;     % nodes per layer
nbrOfEpochs = 400;   % opt. iterations 
W1 = cell(1, node);
W2 = cell(1, node);

% initialize the Ws (network NN weights)
W10_avg = zeros( nbrOfNodes, length(Attributes(1,:)) );
W20_avg = zeros( length(Classifications(1,:)), nbrOfNodes + 1 );
for i = 1:node
    % Initialize matrices with random weights (weights for each layer)
    W1{i} = randn( nbrOfNodes, length(Attributes(1,:)) );
    W2{i} = randn( length(Classifications(1,:)), nbrOfNodes + 1 );
    W10_avg = W10_avg + 1/node*W1{i};
    W20_avg = W20_avg + 1/node*W2{i};
end


V1 = cell(1, node);
V2 = cell(1, node);

for i = 1:node
    V1{i} = zeros(size(W1{i}));
    V2{i} = zeros(size(W2{i}));
end

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

save('opt_data.mat');
fprintf('Opt data has been created!\n');