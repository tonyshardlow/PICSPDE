function bj = get_oned_bjV(dtref,J,a,r,epsilon)
jj  = [ 1:J/2 -J/2+1:-1]'; myeps=0.001;
lambda=epsilon*2*pi*jj.^2/a;
root_qj=[0; abs(jj).^-((2*r+1+myeps)/2)];% set decay for H^r
vr=[dtref;(1-exp(-2*lambda*dtref))/2./lambda];% set variance
bj=root_qj.*sqrt(vr/2/pi)*J;

