function u=turn_band_wm(grid1, grid2, M, q,  ell)
[xx, yy] =ndgrid(grid1, grid2);    
sum=zeros(size(xx(:)));
% choose radius T to contain all grid points
T=norm([norm(grid1,inf),norm(grid2,inf)]);
for j=1:M, 
    theta=j*pi/M; e=[cos(theta); sin(theta)]; % uniformly spaced
    tt=[xx(:), yy(:)]*e;   % project
    [gridt, Z]=squad(2*T, 64, 64, @(s) f(s, q, ell)); 
    Xi=interp1(gridt-T, real(Z), tt); % interpolate 
    sum=sum+Xi; % cumulative sum
end;
u=sum/sqrt(M);
u=reshape(u, length(grid1), length(grid2));

function f=f(s, q, ell) % spectral density
f=gamma(q+1)/gamma(q)*(ell^2*abs(s))/(1+(ell*s)^2)^(q+1);
    
        
            
