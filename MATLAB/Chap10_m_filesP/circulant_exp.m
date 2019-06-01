function [t,X,Y,c]=circulant_exp(N,dt,ell)
t=[0:(N-1)]'*dt;    c=exp(-abs(t)/ell);
[X,Y]=circulant_embed_sample(c);

