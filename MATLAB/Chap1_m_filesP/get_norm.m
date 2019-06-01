function [l2_norm, h1_norm] = get_norm(fhandle, a, b, J)
grid=[a:(b-a)/J:b]';
% evaluate the function on the grid
u=fhandle(grid); 
[Uk,nu]=get_coeffs(u, a, b);
l2_norm =sqrt(b-a)* norm(Uk);
dUk=nu.*Uk; 
h1_norm = norm([l2_norm, sqrt(b-a)*norm(dUk)]);
