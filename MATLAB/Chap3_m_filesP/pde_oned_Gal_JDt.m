function [t,ut]=pde_oned_Gal_JDt(u0,T,a,Nref,kappa,Jref,J,epsilon,...
                                     fhandle)
N=Nref/kappa;
Dt=T/N; t=[0:Dt:T]'; 
% initialise
ut=zeros(Jref+1,N+1);
% use IJJ to set unwanted modes to zero.
IJJ=J/2+1:Jref-J/2-1;  
% set linear operator
lambda=2*pi*[0:Jref/2 -Jref/2+1:-1]'/a; 
M=epsilon*lambda.^2; EE=1./(1+Dt*M); EE(IJJ)=0;
% set initial condition
ut(:,1)=u0; u=u0(1:Jref); uh=fft(u); uh(IJJ)=0; 
for n=1:N, % time loop
    fhu=fft(fhandle(u));    fhu(IJJ)=0;
    uh_new=EE.*(uh+Dt*fhu); % semi-implicit Euler step
    u=real(ifft(uh));    ut(1:Jref,n+1)=u;
    uh=uh_new; uh(IJJ)=0; 
end
ut(Jref+1,:)=ut(1,:); % make periodic

