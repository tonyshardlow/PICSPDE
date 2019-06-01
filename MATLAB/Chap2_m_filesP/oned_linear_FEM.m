function [uh,A,b,K,M]=oned_linear_FEM(ne,p,q,f)
% set-up 1d FE mesh
h=(1/ne); xx=0:h:1; nvtx=length(xx);
J=ne-1; elt2vert=[1:J+1;2:(J+2)]';
% initialise global matrices
K = sparse(nvtx,nvtx); M = sparse(nvtx,nvtx); b=zeros(nvtx,1);      
% compute element matrices 
[Kks,Mks,bks]=get_elt_arrays(h,p,q,f,ne);
% Assemble element arrays into global arrays
for row_no=1:2
    nrow=elt2vert(:,row_no);
    for col_no=1:2
        ncol=elt2vert(:,col_no);
        K=K+sparse(nrow,ncol,Kks(:,row_no,col_no),nvtx,nvtx);
        M=M+sparse(nrow,ncol,Mks(:,row_no,col_no),nvtx,nvtx);
    end
    b = b+sparse(nrow,1,bks(:,row_no),nvtx,1);
end
% impose homogeneous boundary condition
K([1,end],:)=[]; K(:,[1,end])=[]; M([1,end],:)=[]; M(:,[1,end])=[]; 
A=K+M; b(1)=[]; b(end)=[];
% solve linear system for interior degrees of freedom;
u_int=A\b; uh=[0;u_int;0]; plot(xx,uh,'-');



