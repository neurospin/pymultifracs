function [estsingle] = singleBSestimate(EST, estid, param_bs);
% function [estsingle] = singleBSest(EST, estid, param_bs);
% read out single (bootstrap) estimates from n dim estimate obtained by
% MFA_regrest_BS: for BSlimit and BShtest
%
% Herwig Wendt, Lyon, 2006 - 2008

%- check if estimate is available
if estid > length(EST.t)
    error ('Estimate estid not available');
end

%- check which BS estimates are available and necessary
ci_method = param_bs.ci_method;
if find (ci_method == 1); NOR    = 1; else NOR    = 0; end
if find (ci_method == 2); BAS    = 1; else BAS    = 0; end
if find (ci_method == 3); PER    = 1; else PER    = 0; end
if find (ci_method == 4); STU    = 1; else STU    = 0; end
if find (ci_method == 5); BASADJ = 1; else BASADJ = 0; end
if find (ci_method == 6); PERADJ = 1; else PERADJ = 0; end

% write appropriate structure
estsingle.t = EST.t(estid);
estsingle.stdt = EST.BS.stdt(estid);
estsingle.T = EST.BS.T(:, estid)';

if (STU|BASADJ|PERADJ)
    estsingle.stdT = EST.BS.stdT(:, estid)';
end

if BASADJ|PERADJ
    estsingle.TT = EST.BS.TT(:, :, estid)';
end