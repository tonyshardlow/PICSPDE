function [X,Y]=quad_wm(s, N, M, q)
[X,Y]=interp_quad(s, N, M, @(nu) f_wm(nu,q));

function f=f_wm(nu, q) % spectral density 
const=gamma(q+0.5)/(gamma(q)*gamma(0.5));
f=const/((1+nu*nu)^(q+0.5));
   
 



   
 

