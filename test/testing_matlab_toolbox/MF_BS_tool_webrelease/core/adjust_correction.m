function [lgt_corr] = adjust_correction (lgt, j1, j2)

lgt_corr = lgt;

% Leave if the scaling range has not changed
if (j1 == lgt.param_est.j1 && j2 == lgt.param_est.j2) ...
       || lgt.param_est.p == inf ...
       || lgt.param_est.plead_corr == 0
    return
end

ncoef = size (lgt.est, 2);
nj = lgt.nj;
q = lgt.param_est.q;

% Recover the structure function S(p,j):
log_Spj = lgt.log_Spj;

% Recompute \eta(p)
zp_old = lgt.zp;
zp = MFA_BS_regrmat (log_Spj, ones (size (log_Spj)), nj, lgt.param_est.wtype, j1, j2);

% Leave if plead_corr == 2 and zp<0, since the correction should not be used in this case.
if lgt.param_est.plead_corr == 2 && zp < 0
    warning ('zp is negative. I''ll not apply the correction')
    return
end

% Compute old and new correction
log_Spj_corr_old = -1 / lgt.param_est.p * zetacorr_LF (zp_old, 1 : ncoef, lgt.param_est.j1_ini);
log_Spj_corr_new = -1 / lgt.param_est.p * zetacorr_LF (zp, 1 : ncoef, lgt.param_est.j1_ini);

[zid, did, hid, cid] = get_estimate_id (lgt.param_est);
[doz, dod, doc] = which_estimates (lgt.param_est.estimate_select);

% Correct structure functions
if doz
    for iq = 1 : length (zid)
        lgt_corr.est(zid(iq), :) = lgt_corr.est(zid(iq), :) + ...
            (log_Spj_corr_new - log_Spj_corr_old) * q(iq);
    end
end

% Correct spectrum
if dod
    for iq = 1 : length (zid)
        lgt_corr.est(hid(iq), :) = lgt_corr.est(hid(iq), :) + ...
            log_Spj_corr_new - log_Spj_corr_old;
    end
end

% Correct cumulants
if doc
    % The log(2) corrects for the log_2 used in statlog for the cumulants
    lgt_corr.est(cid(1), :) = lgt_corr.est(cid(1), :) + ...
        (log_Spj_corr_new - log_Spj_corr_old) * log(2);
end

lgt_corr.zp = zp;
lgt_corr.param_est.j1 = j1;
lgt_corr.param_est.j2 = j2;

% TODO
% We should also recompute p0, supcoef and mincoef for the new scaling range.
% However, we need the wavelet coefficients to compute this things; we can't
% do it from the lgostat only.
