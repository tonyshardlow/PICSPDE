function [t, u]=EulerMaruyamaTheta(u0,T,N,d,m,fhandle,ghandle,theta)
Dt=T/N; u=zeros(d,N+1); t=[0:Dt:N*Dt]'; sqrtDt=sqrt(Dt);
options=optimset('Display','Off');    
u(:,1)=u0; u_n=u0; % initial data
for n=2:N+1, % time loop
    dW=sqrtDt*randn(m,1); % Brownian increment
    u_explicit=u_n+Dt*fhandle(u_n)+ghandle(u_n)*dW;
    if (theta>0) % solve nonlinear eqns for update
                 % u_explicit is initial guess for nonlinear solve
        v=u_n+(1-theta)*Dt*fhandle(u_n)+ghandle(u_n)*dW;
        u_new=fzero(@(u) -u+v +theta*fhandle(u)*Dt,...
                  u_explicit, options);
    else % explicit case
        u_new=u_explicit;
    end
    u(:,n)=u_new;  u_n=u_new;
end
