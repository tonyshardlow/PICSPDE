function [Jks,invJks,detJks]=get_jac_info(xv,yv,ne,elt2vert);
Jks=zeros(ne,2,2);  invJks=zeros(ne,2,2);
% all vertices of all elements
x1=xv(elt2vert(:,1)); x2=xv(elt2vert(:,2)); x3=xv(elt2vert(:,3));
y1=yv(elt2vert(:,1)); y2=yv(elt2vert(:,2)); y3=yv(elt2vert(:,3));
% Jk matrices,determinants and inverses 
Jks(:,1,1)=x2-x1;Jks(:,1,2)=y2-y1; Jks(:,2,1)=x3-x1;Jks(:,2,2)=y3-y1;
detJks=(Jks(:,1,1).*Jks(:,2,2))-(Jks(:,1,2).*Jks(:,2,1));
invJks(:,1,1)=(1./detJks).*(y3-y1);invJks(:,1,2)=(1./detJks).*(y1-y2); 
invJks(:,2,1)=(1./detJks).*(x1-x3);invJks(:,2,2)=(1./detJks).*(x2-x1);

