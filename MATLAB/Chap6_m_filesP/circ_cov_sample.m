function [X,Y] = circ_cov_sample(c)
N=length(c);    d=ifft(c,'symmetric')*N;
xi=randn(N,2)*[1; sqrt(-1)];
Z=fft((d.^0.5).*xi)/sqrt(N);
X=real(Z); Y=imag(Z);
 
