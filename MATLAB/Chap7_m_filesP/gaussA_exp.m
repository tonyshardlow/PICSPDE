function c=gaussA_exp(x1,x2,a11,a22,a12)
% -2*x1*x2*a12 misprint: gives A with off-diagonal -a12, not +a12.
% Example ex:reduced_exp_cov now uses a12=-0.5 and so is unchanged.
% correction 17-Jul 2026
c=exp(-((x1^2*a11+x2^2*a22)+2*x1*x2*a12));
    
