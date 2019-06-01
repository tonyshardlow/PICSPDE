function [QM]= setseed1(M);
s1 = RandStream('mt19937ar','Seed',1); % create a new stream s1
r1=randn(s1,M,1); % draw M N(0,1) numbers from s1
r0=randn(M,1); % draw M N(0,1) numbers from default stream
QM=cov([r1,r0]); % covariance matrix
