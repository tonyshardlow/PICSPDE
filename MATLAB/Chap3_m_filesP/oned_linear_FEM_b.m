function b=oned_linear_FEM_b(ne,h,f)
nvtx=ne+1; elt2vert=[1:ne; 2:(ne+1)]';
bks=zeros(ne,2); b=zeros(nvtx,1);
bks(:,1) = f(1:end-1,:).*(h/3)+f(2:end,:).*(h/6);
bks(:,2) = f(1:end-1,:).*(h/6)+f(2:end,:).*(h/3);
for row_no=1:2
 nrow=elt2vert(:,row_no);
 b=b+sparse(nrow,1,bks(:,row_no),nvtx,1);
end
b(1,:)=[];b(end,:)=[];
