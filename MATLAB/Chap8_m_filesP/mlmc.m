function [EPu, M]=mlmc(u0,T,d,m,fhandle,ghandle,kappa,epsilon,DTMX)
Levels=ceil(log(2*T/epsilon)/log(kappa))+1; %  (=L+1) 
DT=T*kappa.^(-(0:Levels-1))'; % time steps
L0=find(DT<=DTMX, 1); % coarsest level (=ell_0+1)
M=10*ones(Levels,1); % initial samples
Ckappa=(1+1./kappa); S1=zeros(Levels,1); S2=S1; ML=S1; VL=S1;
for j=L0:Levels
  N=kappa^j;  
  % get samples for level j, initial pass 
  [S1,S2]=getlevel(u0,T,N,d,m,fhandle,ghandle,kappa,M(j),j,L0,S1,S2);
  % estimate variance          
  VL(L0:j)=S2(L0:j)./M(L0:j)-(S1(L0:j)./M(L0:j)).^2; 
  KC=(2/epsilon^2)*sqrt(VL(L0)/DT(L0)); % KC coarse level only
  if j>L0, % esimate samples required (corrections)
    KC=KC+(2/epsilon^2)*sum(sqrt(VL(L0+1:j)./DT(L0+1:j)*Ckappa));
    ML(L0+1:j)=ceil(KC*sqrt(VL(L0+1:j).*DT(L0+1:j)/Ckappa));
  else % estimate sample required (coarsest level)
    ML(L0)=ceil(KC*sqrt(VL(L0)*DT(L0)));
  end
  for l=L0:j, 
    dM=ML(l)-M(l);
    if dM>0 % extra samples needed
      N=kappa^l;   
      M(l)=M(l)+dM; % get dM extra samples
      [S1,S2]=getlevel(u0,T,N,d,m,fhandle,ghandle,kappa,dM,l,L0,S1,S2);
    end
  end  
end
EPu=sum(S1(L0:Levels)./M(L0:Levels));

