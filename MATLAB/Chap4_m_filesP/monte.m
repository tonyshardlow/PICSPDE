function [sample_av, conf95]=monte(samples)
M=length(samples);
conf95 = 2*sqrt(var(samples)/M); sample_av = mean(samples);
