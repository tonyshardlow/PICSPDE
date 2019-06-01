function C_red=reduced_cov(n1,n2,dx1,dx2,fhandle);
C_red=zeros(2*n1-1, 2*n2-1);
for i=1:2*n1-1
    for j=1:2*n2-1
        C_red(i,j)=feval(fhandle, (i-n1)*dx1, (j-n2)*dx2);
    end
end

