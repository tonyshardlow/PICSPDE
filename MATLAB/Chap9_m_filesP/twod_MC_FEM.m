function [mean_u, var_u]=twod_MC_FEM(ns,Q,ell,alpha)
% FEM mesh info
[xv,yv,elt2vert,nvtx,ne,h]=uniform_mesh_info(ns);
xv=reshape(xv,ns+1,ns+1)'; yv=reshape(xv,ns+1,ns+1)';
b_nodes=find((xv==0)|(xv==1)|(yv==0)|(yv==1));
int_nodes=1:nvtx; int_nodes(b_nodes)=[]; 
% specify covariance 
fhandle1=@(x1,x2)gaussA_exp(x1,x2,ell^(-2),ell^(-2),0);
n1=ns+1; n2=ns+1; m1=alpha*n1; m2=alpha*n2;
C_red=reduced_cov(n1+m1,n2+m2,1/ns,1/ns,fhandle1);
% initialise
sum_us=zeros(nvtx,1); sum_sq=zeros(nvtx,1); 
Q2=floor(Q/2); 
for i=1:Q2
    % two realisations of a - with padding
    [z1,z2]=circ_embed_sample_2dB(C_red,n1,n2,m1,m2);
    v1=exp(z1);v2=exp(z2);
    % piecewise constant approximation
    a1=(1/3).*(v1(elt2vert(:,1))+v1(elt2vert(:,2))+v1(elt2vert(:,3)));
    a2=(1/3).*(v2(elt2vert(:,1))+v2(elt2vert(:,2))+v2(elt2vert(:,3)));
    % two realisations of FEM solution & zero bcs
    [u1_int,A1,rhs1]=twod_linear_FEM(ns,xv,yv,elt2vert,...
                                         nvtx,ne,h,a1,ones(ne,1)); 
    [u2_int,A2,rhs2]=twod_linear_FEM(ns,xv,yv,elt2vert,...
                                         nvtx,ne,h,a2,ones(ne,1)); 
    u1=zeros(nvtx,1); u1(int_nodes)=u1_int;
    u2=zeros(nvtx,1); u2(int_nodes)=u2_int;
    sum_us=sum_us+u1+u2; sum_sq=sum_sq+(u1.^2+u2.^2);
end
Q=2*Q2;
mean_u=sum_us./Q; var_u=(1/(Q-1)).*(sum_sq-((sum_us.^2)./Q));



