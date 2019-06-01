function [t,ut]=pde_fd(u0,T,a,N,J,epsilon,fhandle,bctype)
Dt=T/N;   t=[0:Dt:T]'; h=a/J; 
% set matrix A according to boundary conditions
e=ones(J+1,1);    A=spdiags([-e 2*e -e], -1:1, J+1, J+1);
switch lower(bctype)
  case {'dirichlet','d'}
    ind=2:J;   A=A(ind,ind);
  case {'periodic','p'}
    ind=1:J; A=A(ind,ind); A(1,end)=-1; A(end,1)=-1;
  case {'neumann','n'}
    ind=1:J+1; A(1,2)=-2; A(end,end-1)=-2; 
end
EE=speye(length(ind))+Dt*epsilon*A/h^2;
ut=zeros(J+1,length(t)); % initialize vectors 
ut(:,1)=u0; u_n=u0(ind); % set initial condition
for k=1:N, % time loop
  fu=fhandle(u_n);  % evaluate f(u_n)
  u_new=EE\(u_n+Dt*fu); % linear solve for (1+epsilon A)
  ut(ind,k+1)=u_new; u_n=u_new;    
end
if lower(bctype)=='p' | lower(bctype)=='periodic'
    ut(end,:)=ut(1,:); % correct for periodic case
end
