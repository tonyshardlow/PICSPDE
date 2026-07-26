% Example 3.40
% Corrections due to misuse of meshgrid
% TS Dec 2015
T=10; N=1000; a=[2*pi 16]; J=[128 256]; epsilon=1e-3;
x=0:a(1)/J(1):a(1);
y=0:a(2)/J(2):a(2);
[yy,xx]=meshgrid(y,x); % corrected TS Dec 2015
                       
% Initial data as printed in Example 3.40 and in the sentence below Fig 3.5.
% The published Fig 3.5(b) was computed from sin(x2)*cos(pi*x1/8) instead,
% through the meshgrid misuse recorded in the errata at p116; that field jumps
% by 1.78 and 0.29 across the two periodic boundaries, so it cannot have been
% intended. This figure therefore differs from the one printed in the book.
u0=sin(xx).*cos(pi*yy/8);
[t,ut]=pde_twod_Gal(u0,T,a,N,J,epsilon,@(u) u-u.^3);
mesh(x,y,ut(:,:,end)')
xlabel('x_1')
ylabel('x_2')
zlabel('u')
colorbar
