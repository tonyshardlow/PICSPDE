function [mean_u,var_u]=oned_MC_FEM(ne,sigma,mu,P,Q)
h=1/ne; x=[(h/2):h:(1-h/2)]'; 
sum_us=zeros(ne+1,1); sum_sq=zeros(ne+1,1);
for j=1:Q
    xi=-1+2.*rand(P,1); a=mu.*ones(ne,1);
    for i=1:P
        a=a+sigma.*((i.*pi).^(-2)).*cos(pi.*i.*x).*xi(i);
    end
    [u,A,b]=oned_linear_FEM(ne,a,zeros(ne,1),ones(ne,1)); hold on;
    sum_us=sum_us+u; sum_sq=sum_sq+(u.^2);
end
mean_u=sum_us./Q;
var_u=(1/(Q-1)).*(sum_sq-(sum_us.^2./Q)); 

