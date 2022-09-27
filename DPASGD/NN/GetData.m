function [X, y] = GetData(n, name)

    load(name);

    [X, y] = Shufl(X,y);


    X1 = X(1:n,:);
    y1 = y(1:n,:);

    clear X y

    X = X1;
    y = y1;
    
end