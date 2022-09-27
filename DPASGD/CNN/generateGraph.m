function [Lap, W_graph] = generateGraph(n,type)

    if type==1
        % Ring topology
        adj = zeros(n,n);
        for i=2:n-1
            adj(i,i-1) = 1;
            adj(i,i+1) = 1;
        end
        adj (1,2) = 1;adj (1,n) = 1;
        adj (n,n-1) = 1;adj (n,1) = 1;
    end

    if type ==2
        % Line (path) topology
        adj = zeros(n,n);
        for i=2:n-1
            adj(i,i-1) = 1;
            adj(i,i+1) = 1;
        end
        adj (1,2) = 1;
        adj (n,n-1) = 1;
    end

    if type ==3
        % Some random graph
        adj = [0 1 0 0 0 0 0 0 0 0;...
               1 0 1 1 0 0 0 0 0 0;...
               0 1 0 1 0 0 1 0 0 0;...
               0 1 1 0 1 1 0 0 0 0;...
               0 0 0 1 0 1 0 0 0 0;...
               0 0 0 1 1 0 0 0 0 0;...
               0 0 1 0 0 0 0 1 0 0;...
               0 0 0 0 0 0 1 0 1 0;...
               0 0 0 0 0 0 0 1 0 1;...
               0 0 0 0 0 0 0 0 1 0];
    end

        deg = diag(sum(adj,2));         % degree matrix
        Lap = deg - adj;                % Laplacian matrix
        
        m = sum(sum(adj))/2;
        A = zeros(n,m);
        l = 0;

        for i = 1:(n-1)
            for j = i+1:n
                if adj(i,j) > 0.5
                    l = l + 1;
                    A(i,l) =  1;
                    A(j,l) = -1;
                end
            end
        end
        W_graph = mh_matrix(A);
        
end