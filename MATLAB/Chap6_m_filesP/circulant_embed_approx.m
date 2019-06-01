function [X,Y]=circulant_embed_approx(c)
tilde_c=[c; c(end-1:-1:2)]; tilde_N=length(tilde_c);
d=ifft(tilde_c,'symmetric')*tilde_N;
d_minus=max(-d,0); d_pos=max(d,0);
if (max(d_minus)>0)
    disp(sprintf('rho(D_minus)=%0.5g', max(d_minus)));
end;
xi=randn(tilde_N,2)*[1; sqrt(-1)];
Z=fft((d_pos.^0.5).*xi)/sqrt(tilde_N);
N=length(c); X=real(Z(1:N)); Y=imag(Z(1:N));
    
 

    
