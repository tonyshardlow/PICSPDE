function XJ=fkl_1d(J,T,ell)
kk=[0:2*J-1]'; % col vector for range of k
b=sqrt(1/T*exp(-pi*(kk*ell/T).^2));   
b(1)=sqrt(0.5)*b(1);
% complex coefficients: a cosine and an independent sine term for each k, so
% that the covariance is c(s-t) and not (c(s-t)+c(s+t))/2 (Example 6.10)
xi=(rand(2*J,2)*sqrt(12)-sqrt(3))*[1;1i];
a=b.*xi;  XJ=real(fft(a));  XJ=XJ(1:J);

