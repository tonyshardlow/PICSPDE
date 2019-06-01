function [dW1,dW2]=get_twod_dW(bj,kappa,M)
J=size(bj); 
if(kappa==1)
  nnr=randn(J(1),J(2),M);  nnc=randn(J(1),J(2),M);
else
  nnr=squeeze(sum(randn(J(1),J(2),M,kappa),4));
  nnc=squeeze(sum(randn(J(1),J(2),M,kappa),4));
end
nn2=nnr + sqrt(-1)*nnc; TMPHAT=bsxfun(@times,bj,nn2);
tmp=ifft2(TMPHAT); dW1=real(tmp); dW2=imag(tmp);
