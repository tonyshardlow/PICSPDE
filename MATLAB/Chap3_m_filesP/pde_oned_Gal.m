function [t,ut]=pde_oned_Gal(u0,T,a,N,J,epsilon,fhandle)
Dt=T/N; t=[0:Dt:T]'; ut=zeros(J+1,N+1);
% set linear operator
lambda=2*pi*[0:J/2 -J/2+1:-1]'/a;   M= epsilon*lambda.^2;
EE=1./(1+Dt*M); % diagonal of (1+ Dt M)^{-1}
ut(:,1)=u0; u=u0(1:J); uh=fft(u); % set initial condition
for n=1:N, % time loop
    fhu=fft(fhandle(u)); % evaluate fhat(u)
    uh_new=EE.*(uh+Dt*fhu); % semi-implicit Euler step
    u=real(ifft(uh_new));  ut(1:J,n+1)=u; uh=uh_new;    
end
ut(J+1,:)=ut(1,:); % make periodic
