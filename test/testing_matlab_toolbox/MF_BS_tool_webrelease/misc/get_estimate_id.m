function [idz, idd, idh, idc] = get_estimate_id (param_est)
% Gets the id for each type of estimate, according to param_est.estimate_select.
%
%  - param_est.estimate_select: number xyz - what is to be calculated:  
%    * x : zeta(q) [0 or 1]    (scaling exponents)
%    * y : D(h) [0 or 1]       (multifractal spectrum)
%    * z : cp [0 or 1]         (log-cumulants)
%
% February, 2016

idz = [];
idd = [];
idh = [];
idc = [];

[flag_zq, flag_dh, flag_cp] = which_estimates (param_est.estimate_select);
nq = length (param_est.q);
ncum = param_est.cum;

if flag_zq
    idz = 1 : nq;
end

if flag_dh
    idd = (1 : nq) + flag_zq * nq;
    idh = idd + nq;
end

if flag_cp
    idc = (1 : ncum) + (flag_zq + 2 * flag_dh) * nq;
end