function [estBS]=MFA_BS_bootstrap_regrest(logstat, param_est, param_bs)
% function [estDyad]=MFA_BS_regrest(logstat, param_bs, param_est, wtype)
%   calculate bootstrap  estimates from resamples of structure functions
% 
% Precondition: at least single bootstrap is needed
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
    dimcor = logstat.imagedata + 1;
else
    dimcor = 1;
end

%-- Read in Bootstrap Parameters 
if isfield (logstat.BS, 'estBB')
    doB2 = true;
else
    doB2 = false;
end

%===============================================================================
%-  BOOTSTRAP ESTIMATES
%===============================================================================
%-- reshape variance for N-dim regression 
Vest_mat = squeeze (repmat (logstat.BS.Vest, [1 1 param_bs.n_resamp_1]));
Vest_mat = shiftdim (Vest_mat, length (size (Vest_mat)) - 1);

%-- Normalize Cumulants(j) for regression
if CP
    logstat.BS.estB(:, idx_cum, :) = logstat.BS.estB(:, idx_cum, :) * loge;
end

%-- Regressions
[estBS.T] = MFA_BS_regrmat (logstat.BS.estB, Vest_mat.^2, logstat.nj, wtype, j1, j2);

estBS.T = estBS.T';
estBS.T = estBS.T * regcor;

if DH
    estBS.T(:, idx_dq) = estBS.T(:, idx_dq) + dimcor;
end

%-- Bootstrap Standard Deviation
estBS.stdt = std (estBS.T);

%===============================================================================
%-  DOUBLE BOOTSTRAP ESTIMATES
%===============================================================================
if doB2
    %-- reshape variance for n-dim regression 
    Vest_tensor = squeeze (repmat (Vest_mat, [1 1 1 param_bs.n_resamp_2]));
    Vest_tensor = shiftdim (Vest_tensor, ndims (Vest_tensor) - 1);

    %-- Normalize Cumulants(j) for regression
    if CP
        logstat.BS.estBB(:, :, idx_cum, :) = logstat.BS.estBB(:, :, idx_cum, :) * loge;
    end
    %- Regressions
    [estBS.TT] = MFA_BS_regrmat (logstat.BS.estBB, Vest_tensor.^2, logstat.nj, wtype,  j1, j2);

    estBS.TT = permute (estBS.TT, [3 1 2]);
    estBS.TT = estBS.TT * regcor;
    if DH
        estBS.TT(:, :, idx_dq) = estBS.TT(:, :, idx_dq) + dimcor;
    end
    %- Double Bootstrap Standard Deviation
    estBS.stdT = squeeze (std (estBS.TT));
end


