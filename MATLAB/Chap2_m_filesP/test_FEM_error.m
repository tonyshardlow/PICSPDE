function error=test_FEM_error(ne,uh)
h=(1/ne); xx=0:h:1; Ek2=zeros(ne,1); 
u1s=uh(1:end-1); u2s=uh(2:end);
% quadrature weights and points
weights=h.*[1/6;2/3;1/6];  
x_quad=[xx(1:end-1)',[xx(1:end-1)+h/2]',xx(2:end)'];
for i=1:3
    Ek2=Ek2+weights(i).*Ek2_eval(x_quad(:,i),u1s./h,u2s./h);
end
error=sqrt(sum(Ek2));

function Ek2=Ek2_eval(x,u1,u2);
Ek2=(0.5-x+u1-u2).^2;
return

