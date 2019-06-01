function [t,u]=exp_euler(u0,T,N,d,fhandle)
Dt=T/N;            % set time step
u=zeros(d,N+1);    % preallocate solution u
t=[0:Dt:T]';       % make time vector
u(:,1)=u0; u_n=u0; % set inital data
for n=1:N, % time loop
  u_new=u_n+Dt*fhandle(u_n); % explicit Euler step
  u(:,n+1)=u_new; u_n=u_new;
end

