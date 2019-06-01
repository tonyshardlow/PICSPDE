function [S1,S2]=getlevel(u00,T,N,d,m,fhandle,ghandle,...
                              kappa,MS,L,L0,S1,S2)
S(1)=0; S(2)=0; 
Mstep=10000; % compute samples in blocks
for M=1:Mstep:MS
  MM=min(Mstep,MS-M+1);  
  u0=bsxfun(@times,u00,ones(d,MM));
  if L==L0 
    % compute Euler-Maruyama samples on the coarsest level
    [t,u]=EMpath(u0, T, N, d, m,fhandle,ghandle,1,MM);
    u=squeeze(u(:,:,end));
    S(1)=S(1)+sum(phi(u)); 
    S(2)=S(2)+sum(phi(u).^2);
  else % fine levels
    defaultStream = RandStream.getGlobalStream;
    % save state of random number generator
    savedState = defaultStream.State;
    % compute Euler-Maruyama samples 
    [t,uu]=EMpath(u0, T, N, d, m,fhandle,ghandle,1,MM);
    uref=squeeze(uu(:,:,end));
    % reset random number generator
    defaultStream.State = savedState;
    % recompute the same samples with large time step
    [t,uu]=EMpath(u0, T, N, d, m,fhandle,ghandle,kappa,MM);
    u=squeeze(uu(:,:,end)); 
    X=(phi(uref)-phi(u));
    S(1)= S(1)+sum(X); 
    S(2)= S(2)+sum(X.^2);
  end
end  
S1(L)=S1(L)+S(1);
S2(L)=S2(L)+S(2);  

% define the quantity of interest phi
function phiv=phi(v)
phiv=v(end,:);
