function [dt,err]=spde_fem_convDt(u00,T,a,Nref,kappa,ne,epsilon,...
				  fhandle,ghandle,r,M)
dtref=T/Nref; h=a/ne;
defaultStream=RandStream.getGlobalStream;
savedState=defaultStream.State; 
[t,uref,ureft]=spde_fem_MhDt(u00,T,a,Nref,1,ne,1,epsilon,...
                                fhandle,ghandle,r,M);
for i=1:length(kappa)
    dt(i)=kappa(i)*dtref; defaultStream.State=savedState;
    [t,u,ut]=spde_fem_MhDt(u00,T,a,Nref,kappa(i),ne,1,epsilon,...
                         fhandle,ghandle,r,M);
    S(i)=sum(sum((uref-u).^2)*h); 
end
err=sqrt(S./M); 
