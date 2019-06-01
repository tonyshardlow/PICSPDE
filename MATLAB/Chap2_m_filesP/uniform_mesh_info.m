function [xv,yv,elt2vert,nvtx,ne,h]=uniform_mesh_info(ns)
h=1/ns; x=0:h:1; y=x; [xv,yv]=meshgrid(x,y); 
% co-ordinates of vertices
xv=xv'; xv=xv(:); yv=yv'; yv=yv(:);
n2=ns*ns;nvtx=(ns+1)*(ns+1); ne=2*n2;
% global vertex labels of individual elements
elt2vert=zeros(ne,3); vv=reshape(1:nvtx, ns+1,ns+1);
v1=vv(1:ns,1:ns); v2=vv(2:end,1:ns); v3=vv(1:ns,2:end);
elt2vert(1:n2,:)=[v1(:),v2(:),v3(:)];
v1=vv(2:end,2:end); elt2vert(n2+1:end,:)=[v1(:),v3(:),v2(:)];
triplot(elt2vert,xv,yv); axis square; 