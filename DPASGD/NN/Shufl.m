function [Xout, Yout] = Shufl(X, Y)

    orderedArray = [X Y];
    shuffledArray = orderedArray(randperm(size(orderedArray,1)),:);

    nx = length(X(1,:));

    % DATA SETS; demo file
    Xout = shuffledArray(:,1:nx);
    Yout = shuffledArray(:,nx+1:end);
    
end