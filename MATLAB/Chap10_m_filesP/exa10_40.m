% Example 10.40
% Corrections due to misuse of meshgrid
% TS Dec 2015.
%
% J1 = J2 = 128, Dt = 0.01 on [0,10], and M = 1, as printed in Example 10.40
T=10; N=1000; a=[2*pi 16]; J=[128,128];
alpha=0.1; epsilon=1e-3; sigma=0.1; M=1; kappa=1;
x=[0:a(1)/J(1):a(1)]; y=[0:a(2)/J(2):a(2)];
[yy xx]=meshgrid(y,x);  % corrected TS Dec 2015
% Initial data as printed in Example 10.40, u0 = sin(x1) cos(pi x2/8). The
% published figure was computed from something else, through the meshgrid
% misuse recorded in the errata at p442--469; every other candidate tried here
% is discontinuous across the periodic boundaries, so none can have been
% intended. This figure therefore differs from the one printed in the book.
u0=(sin(xx).*cos(pi*yy/8));
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
