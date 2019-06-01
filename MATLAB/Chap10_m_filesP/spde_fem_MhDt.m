function [t,u,ut]=spde_fem_MhDt(u0,T,a,Nref,kappa,neref,L,epsilon,...
			      fhandle,ghandle,r,M)
ne=neref/L; assert(mod(ne,1)==0); h=(a/ne); nvtx=ne+1;
dtref=T/Nref; Dt=kappa*dtref; t=[0:Dt:T]'; 
p=epsilon*ones(ne,1); q=ones(ne,1); f=ones(ne,1);
% set linear operator
[uh,A,b,KK,MM]=oned_linear_FEM(ne,p,q,f);
EE=MM+Dt*KK; ZM=zeros(1,M);
% Get form of noise
bj = get_onedD_bj(dtref,neref,a,r); bj(ne:end)=0; iFspace=0;
% set initial condition
u=repmat(u0,[1,M]);ut=zeros(nvtx,Nref/kappa+1); ut(:,1)=u(:,1);
for k=1:Nref/kappa, % time loop
  dWJ=get_onedD_dW(bj,kappa,iFspace,M);
  dWL=[ZM;dWJ;ZM];  dWL=dWL(1:L:end,:);  
  gdW=ghandle(u).*dWL;  fu=fhandle(u);  
  for m=1:M
    b(:,m)=oned_linear_FEM_b(ne,h,fu(:,m));
    gdw(:,m)=oned_linear_FEM_b(ne,h,gdW(:,m));
  end
  u1=EE\(MM*u(2:end-1,:)+Dt*b+gdw);
  u=[ZM;u1;ZM];  ut(:,k+1)=u(:,M);
end
