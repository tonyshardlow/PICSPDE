function [t,u]=vdp(u0,T,N,alpha,lambda,sigma)
[t, u]=EulerMaruyama(u0, T, N, 2, 1, @(u) vdp_f(u,lambda, alpha),...
                                         @(u) vdp_g(u,sigma));
function f=vdp_f(u, lambda, alpha) % define drift
f=[u(2); -u(2)*(lambda+u(1)^2)+alpha*u(1)-u(1)^3];
function g=vdp_g(u,sigma) % define diffusion
g=[0; sigma*u(1)];

