function X=uniform_sphere
z=-1+2*rand; theta=2*pi*rand; r=sqrt(1-z*z);
X=[r*cos(theta); r*sin(theta); z];
