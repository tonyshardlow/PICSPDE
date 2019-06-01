function [dt,errT]=pde_oned_Gal_convDt(u0,T,a,Nref,kappa,J,epsilon,...
                                           fhandle)
% reference soln 
[t,ureft]=pde_oned_Gal_JDt(u0,T,a,Nref,1,J,J,epsilon,fhandle);
for i=1:length(kappa) % approximations
     [t,ut]=pde_oned_Gal_JDt(u0,T,a,Nref,kappa(i),J,J,epsilon,fhandle);  
  S(i)=sum((ureft(1:end-1,end)-ut(1:end-1,end)).^2)*(a/J);
end
errT=sqrt(S); dtref=T/Nref; dt=kappa*dtref;
