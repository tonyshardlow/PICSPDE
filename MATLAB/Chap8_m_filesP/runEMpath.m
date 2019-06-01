function [rmsErr,t,u,tref,uref]=runEMpath(u0,T,Nref,d,m,...
                                               fhandle,ghandle,kappa,M)
S=0; Mstep=1000; m0=1;
for mm=1:Mstep:M
  MM=min(Mstep,M-mm+1);   u00=u0(:,mm:m0+MM-1);
  defaultStream = RandStream.getGlobalStream;
  savedState = defaultStream.State; 
  [tref, uref]=EMpath(u00, T, Nref, d, m, fhandle, ghandle,1,MM);  
  defaultStream.State = savedState;
  [t, u]=EMpath(u00, T, Nref, d, m,fhandle, ghandle,kappa,MM);
  err=u(:,:,end)-uref(:,:,end);
  S=S+sum(sum(err.*err));  m0=m0+MM;
end
rmsErr=sqrt(S./M);
