function  X=bmotion(t)
X(1)=0; % start at 0
for n=2:length(t),
    dt=t(n)-t(n-1); X(n)=X(n-1)+sqrt(dt)*randn;
end;

