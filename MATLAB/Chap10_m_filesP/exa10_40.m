% Example 10.40
% Corrections due to misuse of meshgrid
% TS Dec 2015.
%
T=10; N=1000; a=[2*pi 16]; J=[128,256];
alpha=0.1; epsilon=1e-3; sigma=0.1; M=1; kappa=1;
x=[0:a(1)/J(1):a(1)]; y=[0:a(2)/J(2):a(2)];
[yy xx]=meshgrid(y,x);  % corrected TS Dec 2015
% initial data printed in book
% u0=(sin(xx).*cos(pi*yy/8));
% initial data as he intended (x and y switched)
% u0=(sin(yy).*cos(pi*xx/8));
% initial data to give figure on page 10.11
u0=(sin(xx*8/pi).*cos(pi^2*yy/64));
%
[t,u,ut]=spde_twod_Gal(u0,T,a,N,kappa,J,epsilon,...
                       @(u)u-u.^3,...
                       @(u)sigma,...
                       alpha,M);

% either contourf(y,x,ut(:,:,end)) or to copy book 
contourf(x,y,ut(:,:,end)') % notice transpose in last argument
xlabel('x_1')
ylabel('x_2')
colorbar
