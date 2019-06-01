function [bar_x, sig95]=pop_monte(M,T,Dt,baru0,epsilon)
u=[];
for j=1:M,
    u0 = baru0+epsilon*(2*rand(1,2)-1); % sample initial data
    u(j,:)=pop_solve(u0,T,Dt); % solve ODE
end;
[bar_x, sig95]=monte(u(:,1))% analyse first component

