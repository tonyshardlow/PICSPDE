function [sr,sr4,pr]=parseed
stream=RandStream.create('mrg32k3a','Seed',0);
N=5; M=6; sr=zeros(N,M); 
for j=1:N, % draw numbers in serial using substreams indexed by loop
  s=stream;  s.Substream=j;  sr(j,1:M)=randn(s,1,M);
end
s.Substream=4; % draw from substream 4 to recreate 4th row
sr4=randn(s,1,M); % draw the same numbers in parallel
matlabpool open % start a worker pool of default size
pr=zeros(N,M); reset(stream); % reset the stream
parfor j=1:N, % do a loop in parallel
  s=stream;  s.Substream=j;  pr(j,:)=randn(s,1,M);
end
matlabpool close % close the worker pool
