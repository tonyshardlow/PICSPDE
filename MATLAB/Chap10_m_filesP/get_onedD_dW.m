function dW=get_onedD_dW(bj,kappa,iFspace,M)
if(kappa==1)
  nn=randn(length(bj),M);
else
  nn=squeeze(sum(randn(length(bj),M,kappa),3));
end
X=bsxfun(@times,bj,nn);
if(iFspace==1)
  dW=X;
else
  dW=dst1(X);
end

