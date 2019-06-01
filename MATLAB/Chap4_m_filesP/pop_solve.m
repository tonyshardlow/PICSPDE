% this file is not printed in the book
function out=pop_solve(u0,T,Dt)
  N=floor(T/Dt)
  [t,u]=exp_euler(u0,T,N,2,@f)
  out=u(:,end)
  
function fu=f(u)
fu=[u(1)*(1-u(2)), u(2)*(u(1)-1)];

