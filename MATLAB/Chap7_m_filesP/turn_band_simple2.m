function v=turn_band_simple2(grid1, grid2, M)
[xx, yy] =ndgrid(grid1, grid2);    sum=zeros(size(xx(:)));
for j=1:M,
    xi=randn(2,1); theta=pi*j/M; % 
    e=[cos(theta); sin(theta)]; tt=[xx(:), yy(:)]*e; % project
    v=sqrt(1/2)*[cos(tt),sin(tt)]*xi; sum=sum+v; % cumulative sum
end;
v=sum/sqrt(M); % compute sample mean for v
v=reshape(v,length(grid1), length(grid2));

        
            
