function[bar_x,sig95]=pop_monte_anti(M,T,Dt,baru0,epsilon)
u=[];
for j=1:M,
    % two solutions of DE with correlated initial condition
    u0 = baru0+epsilon*(2*rand(1,2)-1);    u(j, :)=pop_solve(u0,T,Dt);
    u0 = 2*baru0-u0;     u(j+M, :)=pop_solve(u0,T,Dt);
end;
% average each antithetic pair before calling monte, so that sig95
% reflects the variance reduction; monte(u(:,1)) would treat the 2M
% correlated samples as independent. correction 17-Jul 2026
[bar_x, sig95]=monte((u(1:M,1)+u(M+1:2*M,1))/2)% analyse first component
