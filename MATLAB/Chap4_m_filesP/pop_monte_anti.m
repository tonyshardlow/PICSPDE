function[bar_x,sig95]=pop_monte_anti(M,T,Dt,baru0,epsilon)
u=[];
for j=1:M,
    % two solutions of DE with correlated initial condition
    u0 = baru0+epsilon*(2*rand(1,2)-1);    u(j, :)=pop_solve(u0,T,Dt);
    u0 = 2*baru0-u0;     u(j+M, :)=pop_solve(u0,T,Dt);
end;
[bar_x, sig95]=monte(u(:,1))% analyse first component
