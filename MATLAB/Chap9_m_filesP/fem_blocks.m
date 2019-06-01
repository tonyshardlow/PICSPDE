function [K_mats,KB_mats,f_vecs,wB]=fem_blocks(ns,xv,yv,elt2vert,...
            nvtx,ne,h,mu_a,nu_a,phi_a,mu_f,nu_f,phi_f,P,N)
        
[Jks,invJks,detJks]=get_jac_info(xv,yv,ne,elt2vert);
b_nodes=find((xv==0)|(xv==1)|(yv==0)|(yv==1));
int_nodes=1:nvtx; int_nodes(b_nodes)=[]; 
wB=feval('g_eval',xv(b_nodes),yv(b_nodes));
M=max(P,N);  % total no. variables
for ell=0:M
    if ell==0
        a=mu_a.*ones(ne,1); f=mu_f.*ones(ne,1);
    else
        if ell<=P
            a=sqrt(nu_a(ell))*phi_a(:,ell);
        else
            a=zeros(ne,1);
        end
        if ell<=N
            f=sqrt(nu_f(ell))*phi_f(:,ell);
        else
            f=zeros(ne,1);
        end
    end
    [Aks,bks]=get_elt_arrays2D(xv,yv,invJks,detJks,ne,elt2vert,a,f);
    A_ell = sparse(nvtx,nvtx);  b_ell = zeros(nvtx,1); 
    for row_no=1:3
        nrow=elt2vert(:,row_no);
        for col_no=1:3
           ncol=elt2vert(:,col_no);
           A_ell=A_ell+sparse(nrow,ncol,Aks(:,row_no,col_no),nvtx,nvtx);
        end
        b_ell = b_ell + sparse(nrow,1,bks(:,row_no),nvtx,1);
    end
    f_vecs{ell+1}=b_ell(int_nodes); 
    KB_mats{ell+1}=A_ell(int_nodes,b_nodes); 
    K_mats{ell+1}=A_ell(int_nodes,int_nodes);
end
end
function g=g_eval(x,y)
g=zeros(size(x)); % boundary condition
end

