function bj = get_onedD_bj(dtref,J,a,r)
jj  = [1:J-1]'; myeps=0.001;
root_qj=jj.^-((2*r+1+myeps)/2);% set decay for H^r
bj=root_qj*sqrt(2*dtref/a);
