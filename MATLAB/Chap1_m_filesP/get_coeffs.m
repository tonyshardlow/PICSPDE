function [Uk, nu] = get_coeffs(u, a, b)
J=length(u)-1; h=(b-a)/J; u1=[(u(1)+u(J+1))/2; u(2:J)];
Uk=(h/(b-a))*exp(-2*pi*sqrt(-1)*[0:J-1]'*a/(b-a)).*fft(u1);
assert(mod(J,2)==0); % J must be even
nu=2*pi/(b-a)*[0:J/2,-J/2+1:-1]';


