function [t,u,ut]=spde_oned_Gal_MJDtV(u0,T,a,N,kappa,Jref,J,epsilon,...
                                        fhandle,ghandle,s,M)
dtref=T/N; Dt=kappa*dtref; t=[0:Dt:T]';
% use IJJ to set unwanted modes to zero.
IJJ=J/2+1:Jref-J/2-1;
% Set Lin Operators
kk  = 2*pi*[0:Jref/2 -Jref/2+1:-1]'/a; 
Dx = (1i*kk); A=-epsilon*Dx.^2;
EE=exp(-Dt*A);
EE1=(1-EE)./A; EE1(1)=Dt;
% Get form of noise
iFspace=1; bj = get_oned_bjV(Dt,Jref,a,s,epsilon); bj(IJJ)=0;
% set initial condition
ut(:,1)=u0; u=u0(1:Jref); uh0=fft(u);
uh=repmat(uh0,[1,M]); u=real(ifft(uh));
for k=1:N/kappa, % Time loop
  uh(IJJ,:)=0; 
  fhu=fft(fhandle(u)); fhu(IJJ,:)=0;
  dW=get_oned_dW(bj,kappa,iFspace,M); dW(IJJ,:)=0;
  uh_new=bsxfun(@times,EE,uh)+bsxfun(@times,EE1,fhu)+ghandle(u)*dW;
  uh=uh_new;   u=real(ifft(uh));  ut(1:Jref,k+1)=u(:,M);   
end
ut(Jref+1,:)=ut(1,:); u=[u; u(1,:)]; % periodic

