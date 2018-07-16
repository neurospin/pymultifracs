function [ci_lo, ci_hi] = get_bs_ci_limits (estb, param_bs)
% Gets the lower and upper limits for the empirical confidence intervals, based
% the bootstrap resamples in estb.
%
% Inputs:
%    - estb: multidimensional array (nbs x n2 x n3 ). 
%            First dimension must correspond to the bootstrap resamples.
%
% Outputs:
%    - ci_lo, ci_hi (n2 x n3): limits of confidence intervals
%
% February 2016

alpha = param_bs.alpha;
id_lo = max (1, floor (param_bs.n_resamp_1 * alpha / 2));
id_hi = param_bs.n_resamp_1 + 1 - id_lo;

tmp= sort(estb, 1);

ci_lo = squeeze (tmp (id_lo, :, :));
ci_hi = squeeze (tmp (id_hi, :, :));

