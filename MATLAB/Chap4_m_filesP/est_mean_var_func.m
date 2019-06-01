function [mu_M, sig_sq_M]=est_mean_var_func(mu,sigma,M)
X=randn(M,1); 
X=mu+sigma*X; % generate M samples from N(mu, sigma^2)
mu_M=mean(X);    % estimate mean
sig_sq_M=var(X); % estimate variance
