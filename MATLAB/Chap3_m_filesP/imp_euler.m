function [t,u]=imp_euler(u0,T,N,d,fhandle)
Dt=T/N;            % set time step 
u=zeros(d,N+1);    % preallocate solution u
t=[0:Dt:N*Dt]';    % set time
options=optimset('Display','off');
u(:,1)=u0; u_n=u0; % set initial condition
for n=1:N,         % time loop 
  u_new=fsolve(@(u) impeuler_step(u,u_n,Dt,fhandle),u_n,options);
  u(:,n+1)=u_new;  u_n=u_new;
end
function step=impeuler_step(u,u_n,Dt,fhandle)  
step=u-u_n-Dt*fhandle(u);
return
