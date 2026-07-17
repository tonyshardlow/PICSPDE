function [h,err]=spde_fem_convh(u00,T,a,N,neref,L,epsilon,...
				  fhandle,ghandle,r,M)
href=a/neref;  xref=[0:href:a]';kappa=1;
% reference soln
defaultStream = RandStream.getGlobalStream;
savedState = defaultStream.State; 
[t,uref,ureft]=spde_fem_MhDt(u00,T,a,N,kappa,neref,1,...
                                epsilon,fhandle,ghandle,r,M);
for j=1:length(L)
  h(j)=href*L(j); u0=u00(1:L(j):end);
  defaultStream.State = savedState;
  [t,u,ut]=spde_fem_MhDt(u0,T,a,N,kappa,neref,L(j),...
                           epsilon,fhandle,ghandle,r,M);
  x=[0:h(j):a]'; uinterp=interp1(x,u,xref);
  % sum(sum(uref(:,end,:)-uinterp(:,end,:)).^2)*href misprint: the .^2 sat
  % outside the inner sum, giving (sum_x D_x)^2, and uref/uinterp are nvtx-by-M
  % so (:,end,:) took only sample M while err divides by M. Match the sibling
  % spde_fem_convDt. correction 17-Jul 2026
  S(j)=sum(sum((uref-uinterp).^2))*href;
end
err=sqrt(S/M);


