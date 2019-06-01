function [t,ut]=spde_fd_n_exp(u0,T,a,N,J,epsilon,sigma,ell,fhandle)
Dt=T/N;   t=[0:Dt:T]'; h=a/J; 
% set matrices 
e = ones(J+1,1);    A = spdiags([e -2*e e], -1:1, J+1, J+1);
%  case {'neumann','n'}
ind=1:J+1; A(1,2)=2; A(end,end-1)=2; 
EE=speye(length(ind))-Dt*epsilon*A/h/h;
ut=zeros(J+1,length(t)); % initialize vectors 
ut(:,1)=u0; u_n=u0(ind); % set initial condition
flag=false;
for k=1:N, % time loop
  fu=fhandle(u_n);
  if flag==false, % generate two samples
      [x,dW,dW2]=circulant_exp(length(ind), h, ell); flag=true;
  else % use second sample from last call
      dW=dW2; flag=false; 
  end;
  u_new=EE\(u_n+Dt*fu+sigma*sqrt(Dt)*dW);
  ut(ind,k+1)=u_new; u_n=u_new;    
end

