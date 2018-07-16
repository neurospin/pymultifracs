function [estDyad]=MFA_BS_regrest(logstat, param_est)
% function [estDyad]=MFA_BS_regrest(logstat, param_bs, param_est, wtype)
%   calculate final estimates from structure functions
% 
%   doesn't calculate bootstrap estimates
%
% Herwig Wendt, Lyon, 2006 - 2008

j1    = param_est.j1;
j2    = min (param_est.j2, size (logstat.est, 2));
wtype = param_est.wtype;

loge = log2 (exp (1));
% RFL, 02/2016: is this needed?
try logstat.regrmult
    regcor = logstat.regrmult;
catch
    regcor = 1;
end

%-- Determine estimates
[ZQ, DH, CP] = which_estimates (param_est.estimate_select);
[idx_zq, idx_dq, idx_hq idx_cum] = get_estimate_id (param_est);

if isfield (param_est, 'imagedata')
    dimcor=logstat.imagedata+1;
else
    dimcor=1;
end

% Coefficients
ESTW=logstat.est;
njW=logstat.nj;


%-- Uniform regularity 
varj = ones (size (logstat.supcoef));
[h_min, ~, ~, h_min_aest]= ...
    MFA_BS_regrmat(log2 (logstat.supcoef), varj, logstat.nj,  wtype,  j1, j2);
[h_max, ~, ~, h_max_aest]= ...
    MFA_BS_regrmat (log2 (logstat.mincoef), varj, logstat.nj,  wtype,  j1, j2);

%-- zetaq, dq, hq, cp
var_mat = ones (size (logstat.est));
if CP
    logstat.est(idx_cum, :) = logstat.est(idx_cum,:) * loge; 
end
[estDyad.t, ~, estDyad.Q, estDyad.aest] = ...
    MFA_BS_regrmat (logstat.est, var_mat,logstat.nj, wtype, j1,j2);

estDyad.t = estDyad.t * regcor;

if DH
    estDyad.t(idx_dq) = estDyad.t(idx_dq) + dimcor;
end

if regcor~=1
    estDyad.aest = estDyad.aest - estDyad.t * (log2 (logstat.scale(1)) - 1 / regcor);
end


%-- Pack output
estDyad.LEst = logstat.Lest;
estDyad.p0=logstat.p0.int;
estDyad.p0noint=logstat.p0.noint;
estDyad.h_min=h_min;
estDyad.h_min_aest=h_min_aest;
estDyad.h_max=h_max;
estDyad.h_max_aest=h_max_aest;
estDyad.param_est = param_est;
estDyad.zp = logstat.zp;
