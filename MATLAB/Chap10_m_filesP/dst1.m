function y = dst1(x)
% if x has length N
% y(k)=sum_{j=1}^N x_j sin( pi k j/(N+1))
[n,m]=size(x);
xx=[zeros(1,m);- x;zeros(1,m);flipud(x)]/2;
xxhat=fft(xx);
y=imag(xxhat(2:n+1,:));
