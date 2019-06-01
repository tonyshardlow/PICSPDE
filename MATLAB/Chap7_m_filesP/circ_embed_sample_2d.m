function [u1,u2]=circ_embed_sample_2d(C_red,n1,n2);
N=n1*n2;
% form reduced matrix of BCCB extension of BTTB matrix C
tilde_C_red = zeros(2*n1,2*n2); 
tilde_C_red(2:2*n1,2:2*n2) = C_red;
tilde_C_red = fftshift(tilde_C_red); 
% sample from N(0, tilde_C)
[u1,u2]=circ_cov_sample_2d(tilde_C_red,2*n1,2*n2);
% recover samples from N(0,C)
u1=u1(:); u2=u2(:);
u1=u1(1:end/2);u1=reshape(u1,n1,2*n2);u1=u1(:,1:2:end); 
u2=u2(1:end/2);u2=reshape(u2,n1,2*n2);u2=u2(:,1:2:end);
