function bj = get_oned_bj(dtref,J,a,r)
jj=[1:J/2, -J/2+1:-1]'; myeps=0.001;
root_qj=[0; abs(jj).^-((2*r+1+myeps)/2)];% set decay for H^r
bj=root_qj*sqrt(dtref/a)*J;
