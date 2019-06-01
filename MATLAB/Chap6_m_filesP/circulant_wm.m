function [t, X, Y, c]=circulant_wm( N, M, dt, q)
Ndash=N+M-1; c=zeros(Ndash+1, 1);  t=dt*[0:Ndash]';
c(1)=1; % t=0 is special, due to singularity in Bessel fn
const=2^(q-1)*gamma(q);
for i=2:Ndash+1,
    c(i)=(t(i)^q)*besselk(q, t(i))/const;
end;
[X,Y]=circulant_embed_approx(c);  X=X(1:N); Y=Y(1:N); t=t(1:N);
