function [X,Y]=circ_cov_sample_2d(C_red,n1,n2);
N=n1*n2; 
Lambda=N*ifft2(C_red);
d=Lambda(:); 
d_minus=max(-d,0);
if (max(d_minus)>0)
    disp(sprintf('Invalid covariance;rho(D_minus)=%0.5g',...
        max(d_minus)));
end;
xi=randn(n1,n2)+i.*randn(n1,n2); 
V=(Lambda.^0.5).*xi;
Z=fft2(V)/sqrt(N); Z=Z(:);
X=real(Z); Y=imag(Z);
