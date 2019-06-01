function [dt,errA,errB]=exp_euler_conv(u0,T,Nref,d,fhandle,kappa)
[t,uref]=exp_euler(u0,T,Nref,d,fhandle); % compute reference soln
uTref=uref(:,end); dtref=T/Nref; uTold=uTref; 
for k=1:length(kappa)
  N=Nref/kappa(k); dt(k)=T/N;
  [t,u]=exp_euler(u0,T,N,d,fhandle); % compute approximate soln
  uT=u(:,end);
  errA(k)=norm(uTref-uT); % error by method A
  errB(k)=norm(uTold-uT); uTold=uT; % error by method B
end
