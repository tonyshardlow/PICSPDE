function [h,err]=pde_fem_convh(u00,T,a,N,neref,L,epsilon,fhandle)
href=a/neref; xref=[0:href:a]';kappa=1;
[t,uref]=pde_fem_hDt(u00,T,a,N,kappa,neref,1,epsilon,fhandle);
for j=1:length(L)
    h(j)=href*L(j);  u0=u00(1:L(j):end);
    [t,u]=pde_fem_hDt(u0,T,a,N,kappa,neref,L(j),epsilon,fhandle);
    x=[0:h(j):a]'; uinterp=interp1(x,u,xref);
    S(j)=sum(uref(:,end)-uinterp(:,end)).^2*href;
end
err=sqrt(S);
