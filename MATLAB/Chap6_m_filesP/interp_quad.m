function [X,Y]=interp_quad(s, N, M, fhandle)
T=max(s)-min(s); 
[t, Z]=squad(T, N, M, fhandle);
I=interp1(t+min(s), Z, s); 
X=real(I); Y=imag(I);

    
