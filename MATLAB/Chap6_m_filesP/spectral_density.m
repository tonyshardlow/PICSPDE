function [f, nu]=spectral_density(X, T)
J=length(X)-1;[Uk,nu]=get_coeffs(X, 0,T);
f=abs(Uk).^2*T/(2*pi);

