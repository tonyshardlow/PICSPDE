function X = gauss_chol(mu,C)
R=chol(C); Z=randn(numel(mu),1); X=reshape(mu(:)+R'*Z,size(mu)); % mu row or column
