function v=turn_band_exp_3d(grid1, grid2, grid3, M,  Mpad,  ell)
[xx,yy,zz]=ndgrid(grid1,grid2,grid3); % x,y,z points
sum=zeros(size(xx(:))); % initialise
T=norm(max(abs([grid1,grid2,grid3])));
gridt=-T+(2*T/(M-1))*(0:(M+Mpad-1))';% radius T encloses all points
c=cov(gridt,ell);% evaluate covariance
for j=1:M,
    X=circulant_embed_approx(c); % sample X using Algorithm 6.10
    e=uniform_sphere; %  sample e using Algorithm 4.6
    tt =[xx(:), yy(:), zz(:)]*e; % project
    Xi=interp1(gridt, X, tt); sum=sum+Xi;
end; 
v=sum/sqrt(M); v=reshape(v,length(grid1), length(grid2), length(grid3));

function f=cov(t, ell) % covariance given by turning bands operator
f= (1-t/ell).*exp(-t/ell); 
