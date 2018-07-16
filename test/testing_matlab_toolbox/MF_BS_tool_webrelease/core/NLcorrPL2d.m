function [lead, ZPJCorr, ZPJ] = NLcorrPL2d (coef, lead, param_est)

j1    = param_est.j1;
j2    = param_est.j2;
wtype = param_est.wtype;
j1_ini  = param_est.j1_ini;
plead_corr   = param_est.plead_corr;
p     = param_est.p;

if p ~= inf
    for j = 1 : length (coef.value)
        ZPJ(j) = log2 (mean (abs(coef.value{j}(:)) .^ p));
    end
    zp = MFA_BS_regrmat (ZPJ, ones (size (ZPJ)), coef.nj, wtype, j1, j2);
else
   for j = 1 : length(coef.value)
       ZPJ(j) = log2 (max (abs (coef.value{j}(:))));
   end
   zp = MFA_BS_regrmat (ZPJ, ones (size (ZPJ)), coef.nj, wtype, j1, j2);
   if zp > 0
       zp = inf;
   else
       zp = -inf;
   end
end

warn_msg_1 = [
'\eta(p) < 0. Correction term  applied to p-leaders undefined. ' ...
'Final estimates can be biased. A smaller value of p (or larger value of gamint) should be selected.' ...
];

warn_msg_2 = [...
'\eta(p) < 0. Correction term for p-leaders is undefined and was NOT applied. ' ...
'Final estimates can be biased. A smaller value of p (or larger value of gamint) should be selected.' ...
];


% Use correction if not using leaders, and plead_corr and zp have adequate values
if p == inf
    ZPJCorr = ones (length (coef.value), 1);
else
    if plead_corr == 1 || (plead_corr == 2 && zp > 0)

        ZPJCorr = 2 .^ (-1 / p * zetacorr_LF (zp, 1 : length(coef.value), j1_ini));

        for j=1 : length (lead.value)
            lead.value{j} = lead.value{j} * ZPJCorr(j);
        end

        if plead_corr == 1 && (zp < 0)
            warning (warn_msg_1)
        end

    else   % Do not use correction
        ZPJCorr = NaN (length (coef.value), 1);

        if plead_corr == 2
            warning (warn_msg_2)
        end
    end
end

% Pack estimated zp in lead:
lead.zp = zp;
