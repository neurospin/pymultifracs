function [zqhqcorr] = zetacorr_LF(zp,JJ,J1LF);

JJ0=JJ-J1LF+1;
zqhqcorr = log2( (1-2.^(-JJ0*zp)) / ( 1 - 2^(-zp) ) );