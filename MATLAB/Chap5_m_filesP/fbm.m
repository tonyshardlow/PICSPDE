function X=fbm(t,H)
N=length(t); C_N=[];
for i=1:N, % compute covariance matrix
    for j=1:N,
        ti=t(i); tj=t(j);
        C_N(i,j)=0.5*(ti^(2*H)+tj^(2*H)-abs(ti-tj)^(2*H));
    end;
end;
[U,S]=eig(C_N); xsi=randn(N, 1); X=U*(S^0.5)*xsi;
