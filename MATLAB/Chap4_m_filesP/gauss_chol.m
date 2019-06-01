function X = gauss_chol(mu,C)
R=chol(C); Z=randn(size(mu)); X=mu+R'*Z;
