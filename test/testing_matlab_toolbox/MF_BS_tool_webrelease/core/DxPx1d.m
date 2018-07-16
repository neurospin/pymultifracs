function [coef, leaders] = DxPx1d(data, param_est)
% Compute 1d wavelet wavelet coefficients and wavelet leaders.

data = data(:)';

Nwt    = param_est.nwt;
gamint = param_est.gamint;
p      = param_est.p;
symm   = param_est.sym;
j1_ini   = param_est.j1_ini;
j1     = param_est.j1;
j2     = param_est.j2;
wtype  = param_est.wtype;
Norm   = 1;

%-- Initialize the wavelet filters
n = length (data) ;         % data length
if symm == 0 % Daubechies Wavelet
    h   = rlistcoefdaub (Nwt) ;      % filter
    nl  = length(h) ;                % length of filter, store to manage edge effect later
    gg1 = -1 * (-1) .^ (1:nl) .* h ; % wavelet filter
    hh1 = fliplr (h);                % scaling filter
    % parameter for the centering of the wavelet

    x0 = 2;
    x0Appro = nl;  %2*Nwt;

else % Daubechies Symmetrized Wavelet
    Nwt     = abs (Nwt);
    h       = rlistcoefdaub (Nwt) ;      % filter
    nl      = length (h) ;               % length of filter, store to manage edge effect later
    gg1c    = -1 * (-1) .^ (1:nl) .* h ; % wavelet filter
    hh1c    = fliplr (h);                % scaling filter
    tmp     = conv (h, hh1c) / sqrt (2);
    nu      = -2 * Nwt + 1;
    [v, nv] = tildeurflippeur (tmp, nu);

    [hh1, nh1] = flippeur (tmp, nu) ;
    nl = length (hh1);
    [gg1, ng1] = flippeur (v, nv) ;

    % parameter for the centering of the wavelet
    x0 = 2 * Nwt;
    x0Appro = nl;  %2*Nwt;
end  % symm == 0

%--- Predict the max # of octaves available given Nwt, and take the min with
nbvoies = fix (log2 (length (data)));
nbvoies = min (nbvoies, fix (log2 (n / (nl + 1))));

%--- Compute the WT, calculate statistics
approW  =data;
for j = 1 : nbvoies         % Loop Scales
    %-- Phase 1a: get the wavelet coefficients/appro at this scale
    njtemp = length(approW);

    conv_gg1 = conv(approW, gg1) ;
    conv_hh1 = conv(approW, hh1) ;
    conv_gg1(isnan (conv_gg1)) = Inf;
    conv_hh1(isnan (conv_hh1)) = Inf;

    fp = nl - 1; % index of first good value
    lp = njtemp; % index of last good value

    % replace border with Inf
    conv_gg1(1      : fp - 1) = Inf;
    conv_gg1(lp + 1 : end   ) = Inf;
    conv_hh1(1      : fp - 1) = Inf;
    conv_hh1(lp + 1 : end   ) = Inf;

    %-- centering and decimation
    approW = conv_hh1((1 : 2 : njtemp) + x0Appro - 1);
    decime = conv_gg1((1 : 2 : njtemp) + x0 - 1);

    %-- passage Norme L1
    value = decime * 2 ^ (j / 2) / 2 ^ (j / Norm);

    %-- max before integration
    lesi = find (isfinite (value));

    %-- Quit if there aren't enough values
    if length(lesi) < 3
        break
    end

    %-- fractional integration and max
    value    = value * 2 ^ (gamint * j);
    AbsdqkW  = abs (value);
    all_xpos = 1 : 2^j : n;

    %-- store results
    coef.value{j} = value(lesi);
    coef.xpos{j}  = all_xpos(lesi);
    coef.nj(j)    = length (lesi);    % number of valid coefficients

    if j < j1_ini
        leaders.value{j} = NaN (size (coef.value{j}));
        leaders.xpos{j}  = coef.xpos{j};
        leaders.nj(j)    = coef.nj(j);
    elseif j == j1_ini
        if p == inf    % wavelet leaders
            voisin = max ([AbsdqkW(1 : end - 2) ; ...
                           AbsdqkW(2 : end - 1) ; ...
                           AbsdqkW(3 : end    ) ]);
        else         %p-leaders
            AbsdqkW = 2 ^ j .* AbsdqkW .^ p ;
            voisin  = sum ([AbsdqkW(1 : end - 2) ; ...
                            AbsdqkW(2 : end - 1) ; ...
                            AbsdqkW(3 : end    ) ]);
            voisin = (2 ^ (-j) .* voisin) .^ (1 / p);
        end

        ifini = isfinite(voisin);
        sans_voisin = AbsdqkW; % initial sans_voisin for leaders at next scale:

        %-- store results
        leaders.value{j} = voisin(ifini);
        leaders.xpos{j}  = all_xpos(ifini);
        leaders.nj(j)    = length(voisin);  % number of leaders
    else
        %-- compute current leaders sans and avec voisin
        nc = floor (length (sans_voisin) / 2);
        if p == inf    % wavelet leaders
            sans_voisin = max ([AbsdqkW(1 : nc)             ; ...
                                sans_voisin(1 : 2 : 2 * nc) ; ...
                                sans_voisin(2 : 2 : 2 * nc) ]);
            voisin = max ([sans_voisin(1 : end - 2); ...
                           sans_voisin(2 : end - 1); ...
                           sans_voisin(3 : end    )]);
        else    % p-leaders
            sans_voisin = sum ([2 ^ j .* AbsdqkW(1 : nc) .^ p ; ...
                                sans_voisin(1 : 2 : 2 * nc)   ; ...
                                sans_voisin(2 : 2 : 2 * nc)   ]);
            voisin = sum ([sans_voisin(1 : end - 2) ; ...
                           sans_voisin(2 : end - 1) ; ...
                           sans_voisin(3 : end    ) ]);
            voisin = (2 ^ (-j) .* voisin) .^ (1 / p);
        end

        ifini = isfinite(voisin);

        if all (~ifini)
            break;
        end

        %-- store results
        leaders.value{j} = voisin(ifini);
        leaders.xpos{j} = all_xpos(ifini);
        leaders.nj(j) = length (leaders.value{j}); % number of leaders
    end % if j <=j1_ini
end    % for j=1:nbvoies

% Pack estimation parameters:
coef.gamint = gamint;
coef.Nwt   = Nwt;

leaders.gamint = gamint;
leaders.Nwt   = Nwt;
leaders.p     = p;
% NLcorrPL will pack zp


%-- correct p-leaders for nonlinearity
[leaders] = NLcorrPL (coef, leaders, param_est);
coef.zp   = leaders.zp;
