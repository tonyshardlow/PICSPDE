function [errJ]=pde_oned_Gal_convJ(u0,T,a,N,Jref,J,epsilon, ...
                                   fhandle)
% reference soln
[tref,ureft]=pde_oned_Gal_JDt(u0,T,a,N,1,Jref,Jref,epsilon,fhandle);
for i=1:length(J) % approximation with J=J(i)
  [t,ut]=pde_oned_Gal_JDt(u0,T,a,N,1,Jref,J(i),epsilon,fhandle);  
  S(i)=sum((ureft(1:end-1,end)-ut(1:end-1,end)).^2)*a/Jref;
end
errJ=sqrt(S);
