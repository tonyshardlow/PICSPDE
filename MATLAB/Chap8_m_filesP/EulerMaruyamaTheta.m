function [t, u]=EulerMaruyamaTheta(u0,T,N,d,m,fhandle,ghandle,theta)
Dt=T/N; u=zeros(d,N+1); t=[0:Dt:N*Dt]'; sqrtDt=sqrt(Dt);
u(:,1)=u0; u_n=u0; % initial data
for n=2:N+1, % time loop
    dW=sqrtDt*randn(m,1); % Brownian increment
    u_explicit=u_n+Dt*fhandle(u_n)+ghandle(u_n)*dW;
    if (theta>0) % solve nonlinear eqns for update
                 % u_explicit is initial guess for nonlinear solve
        v=u_n+(1-theta)*Dt*fhandle(u_n)+ghandle(u_n)*dW;
        u_new=newton_theta(fhandle,v,theta*Dt,u_explicit);
    else % explicit case
        u_new=u_explicit;
    end
    u(:,n)=u_new;  u_n=u_new;
end
end

function w=newton_theta(fhandle,v,c,w)
% Newton's method for F(w)=w-v-c*f(w)=0, starting from w, with a
% finite-difference Jacobian of f, so that any dimension d is allowed.
d=numel(w); I=eye(d);
for it=1:50
  fw=fhandle(w); F=w-v-c*fw;
  if norm(F)<=1e-12*(1+norm(w)), return; end
  Jf=zeros(d);
  for k=1:d
    e=sqrt(eps)*max(1,abs(w(k)));
    Jf(:,k)=(fhandle(w+e*I(:,k))-fw)/e;
  end
  w=w-(I-c*Jf)\F;
end
end
