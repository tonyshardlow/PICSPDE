function XJ=fkl_1d(J,T,ell)
kk=[0:2*J-1]'; % col vector for range of k
b=sqrt(1/T*exp(-pi*(kk*ell/T).^2));   
b(1)=sqrt(0.5)*b(1);
xi=rand(2*J,1)*sqrt(12)-sqrt(3); 
a=b.*xi;  XJ=real(fft(a));  XJ=XJ(1:J);

