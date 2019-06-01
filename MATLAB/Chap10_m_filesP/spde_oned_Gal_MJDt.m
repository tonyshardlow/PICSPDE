function [t,u,ut]=spde_oned_Gal_MJDt(u0,T,a,N,kappa,Jref,J,epsilon,...
                                        fhandle,ghandle,r,M)
dtref=T/N; Dt=kappa*dtref; t=[0:Dt:T]';
% use IJJ to set unwanted modes to zero.
IJJ=J/2+1:Jref-J/2-1;  
% set Lin Operators
kk  = 2*pi*[0:Jref/2 -Jref/2+1:-1]'/a; 
Dx = (1i*kk); MM=-epsilon*Dx.^2; EE=1./(1+Dt*MM); EE(IJJ)=0;
% get form of noise
iFspace=1; bj = get_oned_bj(dtref,Jref,a,r); bj(IJJ)=0;
% set initial condition
ut(:,1)=u0; u=u0(1:Jref); uh0=fft(u);
uh=repmat(uh0,[1,M]); u=real(ifft(uh));  
for n=1:N/kappa, % time loop
  uh(IJJ,:)=0; fhu=fft(fhandle(u)); fhu(IJJ,:)=0;
  dW=get_oned_dW(bj,kappa,iFspace,M); dW(IJJ,:)=0;
  gdWh=fft(ghandle(u).*real(ifft(dW))); gdWh(IJJ,:)=0;
  uh_new=bsxfun(@times,EE,uh+Dt*fhu+gdWh);
  uh=uh_new;   u=real(ifft(uh));  ut(1:Jref,n+1)=u(:,M);   
end
ut(Jref+1,:)=ut(1,:); u=[u; u(1,:)]; % periodic
