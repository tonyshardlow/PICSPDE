function [t,ut]=pde_fem(u0,T,a,N,ne,epsilon,fhandle)
h=a/ne; nvtx=ne+1; Dt=T/N; t=[0:Dt:T]'; 
p=epsilon*ones(ne,1);q=ones(ne,1);f=ones(ne,1);
[uh,A,b,KK,MM]=oned_linear_FEM(ne,p,q,f);
EE=(MM+Dt*KK); ZM=0;
ut=zeros(nvtx,N+1); ut(:,1)=u0; u=u0; % set initial condition
for k=1:N, % time loop
  fu=fhandle(u);     b=oned_linear_FEM_b(ne,h,fu);
  u_new=EE\(MM*u(2:end-1,:)+Dt*b);
  u=[ZM;u_new;ZM];   ut(:,k+1)=u;
end
