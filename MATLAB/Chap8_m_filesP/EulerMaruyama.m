function [t, u]=EulerMaruyama(u0,T,N,d,m,fhandle,ghandle)
Dt=T/N; u=zeros(d,N+1); t=[0:Dt:T]'; sqrtDt=sqrt(Dt);  
u(:,1)=u0; u_n=u0; % initial data 
for n=1:N, % time loop
    dW=sqrtDt*randn(m,1); % Brownian increment
    u_new=u_n+Dt*fhandle(u_n)+ghandle(u_n)*dW;  
    u(:,n+1)=u_new; u_n=u_new;
end
