function [out]=l2_sq_mct(T,a,N,J,M,epsilon,sigma)
v=0; u0=zeros(J+1,1);
parfor i=1:M,
    [t,ut]=spde_fd_d_white(u0,T,a,N,J,epsilon,sigma,@(u) 0);
    v=v+sum(ut(1:end-1,end).^2);
end;
out= v*a/J/M;

