function [u_int,A_int,rhs]=twod_linear_FEM(ns,xv,yv,elt2vert,...
                                                nvtx,ne,h,a,f)
[Jks,invJks,detJks]=get_jac_info(xv,yv,ne,elt2vert);
[Aks,bks]=get_elt_arrays2D(xv,yv,invJks,detJks,ne,elt2vert,a,f);
A = sparse(nvtx,nvtx); b = zeros(nvtx,1);      
for row_no=1:3
    nrow=elt2vert(:,row_no);
    for col_no=1:3
        ncol=elt2vert(:,col_no);
        A=A+sparse(nrow,ncol,Aks(:,row_no,col_no),nvtx,nvtx);
    end
    b = b+sparse(nrow,1,bks(:,row_no),nvtx,1);
end
% get discrete Dirichlet boundary data 
b_nodes=find((xv==0)|(xv==1)|(yv==0)|(yv==1));
int_nodes=1:nvtx; int_nodes(b_nodes)=[]; b_int=b(int_nodes);
wB=feval('g_eval',xv(b_nodes),yv(b_nodes));
% solve linear system for interior nodes;
A_int=A(int_nodes,int_nodes); rhs=(b_int-A(int_nodes,b_nodes)*wB);
u_int=A_int\rhs; 
uh=zeros(nvtx,1); uh(int_nodes)=u_int; uh(b_nodes)=wB;
m=ns+1;mesh(reshape(xv,m,m),reshape(yv,m,m),reshape(uh,m,m)); 
axis square; title('finite element solution');
end
function g=g_eval(x,y)
g=zeros(size(x));
end


