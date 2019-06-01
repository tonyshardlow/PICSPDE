function dW=get_oned_dW(bj,kappa,iFspace,M)
J=length(bj);
if(kappa==1)
  nn=randn(J,M);
else
  nn=squeeze(sum(randn(J,M,kappa),3));
end
nn2=[nn(1,:);(nn(2:J/2,:)+1i*nn(J/2+2:J,:))/sqrt(2);...
     nn(J/2+1,:);(nn(J/2:-1:2,:)-1i*nn(J:-1:J/2+2,:))/sqrt(2)];
X= bsxfun(@times,bj,nn2);
if(iFspace==1)
  dW=X;
else
  dW=real(ifft(X));
end
