function [t,ut]=pde_twod_Gal(u0,T,a,N,J,epsilon,fhandle)
Dt=T/N; t=[0:Dt:T]'; ut=zeros(J(1)+1,J(2)+1,N);
% set linear operators
lambdax=2*pi*[0:J(1)/2 -J(1)/2+1:-1]/a(1);
lambday=2*pi*[0:J(2)/2 -J(2)/2+1:-1]/a(2);
[lambdayy lambdaxx]=meshgrid(lambday,lambdax); % corrected TS
                                               % Dec 2015
M=epsilon*(lambdaxx.^2+lambdayy.^2); EE=1./(1+Dt*M);
ut(:,:,1)=u0; u=u0(1:J(1),1:J(2)); uh=fft2(u); % set initial data 
for n=1:N, % time loop
  fhu=fft2(fhandle(u)); % compute fhat
  uh_new=EE.*(uh+Dt*fhu);
  u=real(ifft2(uh_new)); ut(1:J(1), 1:J(2), n+1)=u;   uh=uh_new;
end
ut(end,:,:)=ut(1,:,:); ut(:,end,:)=ut(:,1,:); % make periodic
