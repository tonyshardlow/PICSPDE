function [X, M]=reject_uniform
M=0; X=[1;1]; % make sure initial X is rejected
while norm(X)>1, % reject
    M=M+1; X=2*rand(1,2)-1; % generate sample
end;



