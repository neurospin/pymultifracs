function [logstat] = MFA_BS_compute_logstat (coef, param)
% function [logstat] = MFA_BS_compute_structure_functions (coef, param_est)
%
% Computes structure functions and cumulants.
%
% Inputs:
%    - coef: structure with fields:
%        * coef.value: {1 x nj} cell array with coefficients for each scale
%          Computations are always performed on coef.value{j}(:)
%    - param_est: struct with estimation parameters.
%
% Output:
%    - logstat: structure with fields:
%        *
%
% February, 2016

num_scales = length (coef.value);
fhandle = param.fhandle;

for j = 1 : length (coef.value)  % loop scales

    sample = abs (coef.value{j}(:));

    [logsf, fdq, fhq, cum] = fhandle (sample, param);

    % Some of the outputs might be empty depending on param.estimate_select
    logstat.est(:, j) = [logsf(:) ; fdq(:) ; fhq(:) ; cum(:)];
    logstat.supcoef(j) = max (abs (sample));
    logstat.mincoef(j) = min (abs (sample));
end

logstat.nj = coef.nj;
logstat.Lest = [length(logsf) , length(fdq) , length(fhq) , length(cum)];
logstat.p0 = p0est (coef, param);
logstat.scale = 2 .^ (1 : num_scales);
logstat.param_est = param;
if param.j2 > num_scales
    logstat.param_est.j2 = num_scales;
end
logstat.zp = coef.zp;

% Recompute correction to store log_Spj in logstat
% RFL: I repeat this computation to keep the code cleaner and avoid adding
% a weird output to DxPx?d
if ~isfield ('coef', 'p')
    if coef.imagedata
        [~, ~, log_Spj] = NLcorrPL2d (coef, coef, param);
    else
        [~, ~, log_Spj] = NLcorrPL (coef, coef, param);
    end
end
logstat.log_Spj = log_Spj;
