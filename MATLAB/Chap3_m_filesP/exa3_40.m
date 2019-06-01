% Example 3.40
% Corrections due to misuse of meshgrid
% TS Dec 2015
T=10; N=1000; a=[2*pi 16]; J=[128 256]; epsilon=1e-3;
x=0:a(1)/J(1):a(1);
y=0:a(2)/J(2):a(2);
[yy,xx]=meshgrid(y,x); % corrected TS Dec 2015
                       
% initial data as printed in book
%u0=sin(xx).*cos(pi*yy/8);
% initial data gives Fig 3.5b
u0=sin(yy).*cos(pi*xx/8);
[t,ut]=pde_twod_Gal(u0,T,a,N,J,epsilon,@(u) u-u.^3);
mesh(x,y,ut(:,:,end)')
xlabel('x_1')
ylabel('x_2')
zlabel('u')
colorbar
