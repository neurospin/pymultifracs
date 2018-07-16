function [zq, dh, cp] = which_estimates (fun)
% Determines which estimates are computed according to be digits of fun:
%
%  - estimate_select: number xyz - what is to be calculated:  
%    * x : zeta(q) [0 or 1]    (scaling exponents)
%    * y : D(h) [0 or 1]       (multifractal spectrum)
%    * z : cp [0 or 1]         (log-cumulants)
%
% February, 2016

zq = false;
dh = false;
cp = false;

if fun >= 100
    zq = true;
    fun = fun - 100;
end
if fun >= 10
    dh = true;
    fun = fun - 10;
end
if fun == 1
    cp = true;
end