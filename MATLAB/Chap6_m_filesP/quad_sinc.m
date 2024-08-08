function Z=quad_sinc(t,  J,  ell)
R=pi/ell;   nustep=2*R/J;  
Z=exp(-sqrt(-1)*t*R)*randn(1,2)*[sqrt(-1);1]/sqrt(2);
for j=1:J-2,
    nu=-R+j*nustep;
    Z=Z+exp(sqrt(-1)*t*nu)*randn(1,2)*[sqrt(-1);1];
end;
Z=Z+exp(sqrt(-1)*t*R)*randn(1,2)*[sqrt(-1);1]/sqrt(2)
% Z=Z*sqrt(ell/(2*pi)); misprint
Z=Z*sqrt(1/(J-1)); correction 7-August 2024
