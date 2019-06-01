function [t,ut]=spde_fd_d_white(u0,T,a,N,J,epsilon,sigma,fhandle)
Dt=T/N;   t=[0:Dt:T]'; h=a/J; 
% set matrices 
e = ones(J+1,1);    A = spdiags([e -2*e e], -1:1, J+1, J+1);
%case {'dirichlet','d'}
ind=2:J;   A=A(ind,ind);
EE=speye(length(ind))-Dt*epsilon*A/h/h;
ut=zeros(J+1,length(t)); % initialize vectors 
ut(:,1)=u0; u_n=u0(ind); % set initial condition
for k=1:N, % time loop
  fu=fhandle(u_n); Wn=sqrt(Dt/h)*randn(J-1,1);
  u_new=EE\(u_n+Dt*fu+sigma*Wn);
  ut(ind,k+1)=u_new; u_n=u_new;    
end

