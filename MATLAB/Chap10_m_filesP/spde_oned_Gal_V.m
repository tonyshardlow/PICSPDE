function [t,u,ut]=spde_oned_Gal_V(u0,T,a,N,J,epsilon,...
                                        fhandle,ghandle,r,M)
Dt=T/N; t=[0:Dt:T]';
% Set Lin Operators
kk  = 2*pi*[0:J/2 -J/2+1:-1]'/a; Dx = (1i*kk); A=-epsilon*Dx.^2;
EE=exp(-Dt*A); EE1=(1-EE)./A; EE1(1)=Dt;
% Get form of noise
iFspace=1; bj = get_oned_bjV(Dt,J,a,r,epsilon); 
% set initial condition
ut(:,1)=u0; u=u0(1:J); uh0=fft(u); 
uh=repmat(uh0,[1,M]); u=real(ifft(uh));
for n=1:N, % Time loop
  fhu=fft(fhandle(u));   dW=get_oned_dW(bj,1,iFspace,M); 
  uh_new=bsxfun(@times,EE,uh)+bsxfun(@times,EE1,fhu)+ghandle(u)*dW;
  uh=uh_new;   u=real(ifft(uh));  ut(1:J,n+1)=u(:,M);   
end
ut(J+1,:)=ut(1,:); u=[u; u(1,:)]; % periodic

