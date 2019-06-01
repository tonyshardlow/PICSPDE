function [u1,u2]=circ_embed_sample_2dB(C_red,n1,n2,m1,m2);
nn1=n1+m1;nn2=n2+m2;N=nn1*nn2;
% form reduced matrix of BCCB extension of BTTB matrix C*
tilde_C_red = zeros(2*nn1,2*nn2); 
tilde_C_red(2:2*nn1,2:2*nn2) = C_red;
tilde_C_red = fftshift(tilde_C_red); 
% sample from N(0, tilde_C)
[u1,u2]=circ_cov_sample_2d(tilde_C_red,2*nn1,2*nn2);
% recover samples from N(0,C)
u1=u1(:); u2=u2(:);
u1=u1(1:2*nn1*n2);u1=reshape(u1,nn1,2*n2);u1=u1(1:n1,1:2:end); 
u2=u2(1:2*nn1*n2);u2=reshape(u2,nn1,2*n2);u2=u2(1:n1,1:2:end);
return
