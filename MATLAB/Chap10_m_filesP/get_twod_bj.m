function bj=get_twod_bj(dtref,J,a,alpha)
lambdax= 2*pi*[0:J(1)/2 -J(1)/2+1:-1]'/a(1);
lambday= 2*pi*[0:J(2)/2 -J(2)/2+1:-1]'/a(2);
% corrected TS Dec 2015
[lambdayy lambdaxx]=meshgrid(lambday,lambdax);
root_qj=exp(-alpha*(lambdaxx.^2+lambdayy.^2)/2);   % smooth noise
bj=root_qj*sqrt(dtref)*J(1)*J(2)/sqrt(a(1)*a(2)); 


  


