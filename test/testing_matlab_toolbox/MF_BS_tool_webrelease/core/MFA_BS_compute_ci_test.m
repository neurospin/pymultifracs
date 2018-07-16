function [stat] = MFA_BS_compute_ci_test (est, param_bs, param_test)

Nest = length(est.LEst);

[~, ~, CP] = which_estimates (est.param_est.estimate_select);
[idx_zq, idx_dq, idx_hq idx_cum] = get_estimate_id (est.param_est);

%- Process estimates one at a time

%- Do confidence intervals for all available estimates
if param_bs.CI
    for estid = 1 : length (est.t);
        [est_single] = singleBSestimate (est, estid, param_bs);

        [stat.confidence{estid}] = BS_CI (est_single, param_bs);
    end
end

% Do tests only for cumulants
if param_bs.TEST && CP
    for estid = 1 : length (idx_cum);
        [est_single] = singleBSestimate (est, idx_cum(estid), param_bs);

        [stat.significance{estid}]=BS_HT (est_single, param_bs, ...
                                          struct ('null_type', param_test.null_type, ...
                                                  'null_param', param_test.null_param(estid)));
    end % Loop through estimates
end

stat.param_bs   = param_bs;
stat.param_test = param_test;

% FIXME: RFL 2016-02-11
% I'm mixing the interfaces here.
% singleBSestimate receives est with the new interface (i.e. with all the 
% bootstrap stuff inside of est.BS) but returns it with the old interface
% (i.e. with all the bootstrap stuff (TT, stdT and the like) directly as 
% fields of est_single).
% I should modify BS_CI to the new interface to standardize everything.