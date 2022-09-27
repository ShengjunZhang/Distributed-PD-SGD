function [opttheta, cost_e, time] = cnn_m_SGD(funObj, cnn_distributed, training, Graph, n_agents, options)
% Runs stochastic gradient descent with momentum to optimize the
% parameters for the given objective.
%
% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  cnn_distributed -  unrolled parameter vector
%  training        -  stores images and labels.
%  options    -  struct to store specific options for optimization
%
% Returns:
%  opttheta   -  optimized parameter vector
%
% Options (* required)
%  epochs*     - number of epochs through data
%  alpha*      - initial learning rate
%  minibatch*  - size of minibatch
%  momentum    - momentum constant, defualts to 0


%%======================================================================
%% Setup
epochs = options.epochs;
gamma = options.gamma;
beta = options.beta;
minibatch = options.minibatch;

% Get m

m = length(training.label{1}); % training set size

% Initilize the parameter matrix

grad_matrix = zeros(n_agents, length(cnn_distributed{1}));
theta_matrix = zeros(n_agents, length(cnn_distributed{1}));
u = zeros(n_agents, length(cnn_distributed{1}));

for i = 1:n_agents
    theta_matrix(i, :) = cnn_distributed{i};
end

%%======================================================================
%% Update loop
cost_e = [];
it = 0;
fprintf('Starting Momentum SGD.\n');
tic;
for e = 1:epochs
    
    rp = randperm(m);

    for s = 1:minibatch:(m-minibatch+1)
            
        
            it = it + 1;
            
            for i = 1:n_agents
                theta_opt = theta_matrix(i, :)';                % get each agent parameters from matrix theta
                mb_data = training.image{i};                    % get training data image
                mb_labels = training.label{i};                  % get training label
                mb_data = mb_data(:,:,:,rp(s:s+minibatch-1));   % randomly pick a data point
                mb_labels = mb_labels(rp(s:s+minibatch-1));     % corresponding label
                
                % evaluate the objective function on the next minibatch for
                % each agent
                
                [temp_cost(i), temp_grad] = funObj(theta_opt,mb_data,mb_labels);
                grad_matrix(i, :) = temp_grad;                  % store gradient in matrix form
            end
            
            u = beta * u + grad_matrix;
            theta_matrix = theta_matrix - gamma*u;

            
            cost = sum(temp_cost)/n_agents;
            cost_e(e, it) = cost;
            
            fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);

    end
    u = Graph*u;
    theta_matrix = Graph*theta_matrix;
    opttheta(e, :) =  sum(theta_matrix,1)/n_agents;
end
% opttheta =  sum(theta_matrix,1)/n_agents;
time = toc;
fprintf('Momentum SGD done, and the running time is %d s.\n', time);
end
