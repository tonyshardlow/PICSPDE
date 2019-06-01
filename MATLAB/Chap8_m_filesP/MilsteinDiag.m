function [t, u]=MilsteinDiag(u0,T,N,d,m,fhandle,ghandle,dghandle)
Dt=T/N; u=zeros(d,N+1); t=[0:Dt:T]';  sqrtDt=sqrt(Dt);
u(:,1)=u0; u_n=u0; % initial data
for n=1:N, % time loop
  dW=sqrtDt*randn(m,1); gu_n=ghandle(u_n);
  u_new=u_n+Dt*fhandle(u_n)+gu_n.*dW ...
         	+0.5*(dghandle(u_n).*gu_n).*(dW.^2-Dt);
  u(:,n+1)=u_new; u_n=u_new;
end
