function [t, u]=EMpath(u0,T,N,d,m,fhandle,ghandle,kappa0,M)
Dtref=T/N;   % small step
Dt=kappa0*Dtref;  % large step 
NN=N/kappa0;   u=zeros(d,M,NN+1); t=zeros(NN+1,1);
gdW=zeros(d,M); sqrtDtref=sqrt(Dtref); u_n=u0;
for n=1:NN+1
  t(n)=(n-1)*Dt;  u(:,:,n)=u_n;   
  dW=sqrtDtref*squeeze(sum(randn(m,M,kappa0),3));
  for mm=1:M
    gdW(:,mm)=ghandle(u_n(:,mm))*dW(:,mm);
  end  
  u_new=u_n+Dt*fhandle(u_n)+gdW;   u_n=u_new;
end
