function [t,u,ut]=spde_twod_Gal(u0,T,a,N,kappa,J,epsilon,...
                                    fhandle,ghandle,alpha,M)
dtref=T/N; Dt=kappa*dtref; t=[0:Dt:T]'; ut=zeros(J(1)+1,J(2)+1,N);
% Set Lin Operator
lambdax= 2*pi*[0:J(1)/2 -J(1)/2+1:-1]'/a(1);
lambday= 2*pi*[0:J(2)/2 -J(2)/2+1:-1]'/a(2);
[lambdayy lambdaxx]=meshgrid(lambday,lambdax); % corrected TS 2015
Dx = (1i*lambdaxx); Dy = (1i*lambdayy);  
A=-( Dx.^2+Dy.^2); MM=epsilon*A; EE=1./(1+Dt*MM);
bj=get_twod_bj(dtref,J,a,alpha);% Get form of noise
u=repmat(u0(1:J(1),1:J(2)),[1,1,M]);% initial condition
uh=repmat(fft2(u0(1:J(1),1:J(2))),[1,1,M]);
% Initialize 
uh1=zeros(J(1),J(2),M); ut=zeros(J(1)+1,J(2)+1,N);ut(:,:,1)=u0;
for n=1:N/kappa, % time loop
  fh=fft2(fhandle(u));  dW=get_twod_dW(bj,kappa,M);
  gudWh=fft2(ghandle(u).*dW); 
  uh_new= bsxfun(@times,EE,(uh+Dt*fh+gudWh));
  u=real(ifft2(uh_new));  ut(1:J(1),1:J(2),n+1)=u(:,:,end);
  uh=uh_new;    
end
u(J(1)+1,:,:)=u(1,:,:); u(:,J(2)+1,:)=u(:,1,:); % make periodic
ut(J(1)+1,:,:)=ut(1,:,:); ut(:,J(2)+1,:)=ut(:,1,:); 

