function [r0,r00] = setseed0(M);
% create a stream s0 (here equivalent to restarting  matlab)
s0 = RandStream('mt19937ar','Seed',0)
% set the default stream to be s0
RandStream.setGlobalStream(s0);
% draw M  N(0,1) numbers from s0
r0=randn(M,1);
% Return to the start of s0
reset(s0);
% draw the same M N(0,1) numbers from s0
r00=randn(M,1);

