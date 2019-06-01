function [t,ut]=pde_fem_hDt(u0,T,a,Nref,kappa,neref,L,epsilon,...
	                 	       fhandle)
ne=neref/L; assert(mod(ne,1)==0); h=a/ne; nvtx=ne+1; 
dtref=T/Nref; Dt=kappa*dtref; t=[0:Dt:T]'; 
p=epsilon*ones(ne,1);q=ones(ne,1);f=ones(ne,1);
[uh,A,b,K,M]=oned_linear_FEM(ne,p,q,f); EE=M+Dt*K; ZM=0;
ut=zeros(nvtx,Nref/kappa+1); ut(:,1)=u0; u=u0;% set initial condition
for k=1:Nref/kappa,% time loop
  fu=fhandle(u);    b=oned_linear_FEM_b(ne,h,fu);
  u_new=EE\(M*u(2:end-1,:)+Dt*b);
  u=[ZM;u_new;ZM]; ut(:,k+1)=u;
end
