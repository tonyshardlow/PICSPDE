function v=turn_band_simple(grid1, grid2)
theta=2*pi*rand;  e=[cos(theta); sin(theta)]; % sample e
[xx,yy]=ndgrid(grid1, grid2); tt=[xx(:), yy(:)] * e; % project
xi=randn(2,1);  v=sqrt(1/2)*[cos(tt), sin(tt)]*xi; % sample v
v =reshape(v, length(grid1), length(grid2));
        
            
