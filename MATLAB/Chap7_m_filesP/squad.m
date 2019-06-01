function [t, Z]=squad(T, N, M, fhandle)
dt=T/(N-1); t=dt*[0:N-1]';
R=pi/dt; dnu=2*pi/(N*dt*M); 
Z=zeros(N, 1);  coeff=zeros(N,1);
for m=1:M,
    for k=1:N,
        nu=-R+((k-1)*M+(m-1))*dnu; 
        xi=randn(1,2)*[1;sqrt(-1)];
        coeff(k)=sqrt(fhandle(nu)*dnu)*xi;
        if ((m==1 && k==1) || (m==M && k==N))
            coeff(k)=coeff(k)/sqrt(2);
        end;
    end;
    Zi=N*ifft(coeff);   
    Z=Z+exp(sqrt(-1)*(-R+(m-1)*dnu)*t).*Zi;   
end;

   
