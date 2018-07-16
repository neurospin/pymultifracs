function [coef, leader] = DxPx2d(data, param_est)
% % Compute 1d wavelet wavelet coefficients and wavelet leaders.
%
% SR, ens-Lyon, 11/2013


Nwt    = param_est.nwt;
gamint = param_est.gamint;
p      = param_est.p;
symm   = param_est.sym;
j1_ini   = param_est.j1_ini;
ndir   = 3;  % number of directions for wavelet coefficients
Norm   = 1;

%-- Initialize the wavelet filters
n = min (size (data));

if symm == 0  % Daubechies Wavelet
    h   = rlistcoefdaub (Nwt) ;        % filter
    nl  = length (h) ;                 % length of filter, store to manage edge effect later
    gg1 = -1 * (-1) .^ (1 : nl) .* h ; % wavelet filter
    hh1 = fliplr(h);                   % scaling filter

    x0 = 2;  % parameter for the centering of the wavelet:
    x0Appro = 2 * Nwt;
else % Daubechies Symmetrized Wavelet
    Nwt     = abs (Nwt);
    h       = rlistcoefdaub (Nwt) ;        % filter
    nl      = length (h) ;                 % length of filter, store to manage edge effect later
    gg1c    = -1 * (-1) .^ (1 : nl) .* h ; % wavelet filter
    hh1c    = fliplr (h);                  % scaling filter
    tmp     = conv (h, hh1c) / sqrt (2);
    nu      = -2 * Nwt + 1;
    [v, nv] = tildeurflippeur (tmp, nu);

    [hh1, nh1] = flippeur (tmp, nu);
    [gg1, ng1] = flippeur (v  , nv);
    nl = length (hh1);

    x0 = 2 * Nwt; % parameter for the centering of the wavelet
    x0Appro = 2 * Nwt;
end

%--- Predict the max # of octaves available given Nwt, and threshold
%--- according to filter length
nbvoies = fix (log2 (min (size (data))));
nbvoies = min (fix (log2 (n / (nl + 3))), nbvoies); % safer, casadestime having problems

%--- Compute the WT, calculate statistics
LL = data;
sidata = size (data);

for j = 1 : nbvoies         % Loop Scales
    njtemp = size(LL);

    %-- border effect
    fp = nl;     % index of first good value
    lp = njtemp; % index of last good value

    %-- OH convolution and subsampling
    OH = conv2(LL, gg1); OH(isnan (OH)) = Inf;
    OH(:, 1         : fp - 1) = Inf;
    OH(:, lp(2) + 1 : end   ) = Inf;
    OH = OH(:, (1 : 2 : njtemp(2)) + x0 - 1);

    %-- HH convolution and subsampling
    HH = conv2(OH, gg1'); HH(isnan (HH)) = Inf;
    HH(1         : fp - 1 , :) = Inf;
    HH(lp(1) + 1 : end    , :) = Inf;
    HH = HH((1 : 2 : njtemp(1)) + x0 - 1, :);

    %-- LH convolution and subsampling
    LH = conv2 (OH, hh1'); LH(isnan (LH)) = Inf;
    LH(1         : fp - 1, :) = Inf;
    LH(lp(1) + 1 : end   , :) = Inf;
    LH = LH((1 : 2 : njtemp(1)) + x0Appro - 1, :);

    clear OH

    %-- OL convolution and subsampling
    OL = conv2 (LL, hh1); OL(isnan (OL)) = Inf;
    OL(:, 1         : fp - 1) = Inf;
    OL(:, lp(2) + 1 : end   ) = Inf;
    OL = OL(:, (1 : 2 : njtemp(2)) + x0Appro - 1);

    %-- HL convolution and subsampling
    HL = conv2 (OL, gg1'); HL(isnan (HL)) = Inf;
    HL(1         : fp - 1, :) = Inf;
    HL(lp(1) + 1 : end   , :) = Inf;
    HL=HL((1 : 2 : njtemp(1)) + x0 - 1, :);

    %-- LL convolution and subsampling
    LL = conv2 (OL, hh1'); LL(isnan (LL)) = Inf;
    LL(1         : fp - 1 , :) = Inf;
    LL(lp(1) + 1 : end    , :) = Inf;
    LL = LL((1 : 2 : njtemp(1)) + x0Appro - 1, :);
    clear OL

    %-- passage Norme L1
    ALH = LH / 2 ^(j / Norm);
    AHL = HL / 2 ^(j / Norm);
    AHH = HH / 2 ^(j / Norm);

    %-- fractional integration by gamma
    ALH = ALH * 2 ^ (gamint * j);
    AHL = AHL * 2 ^ (gamint * j);
    AHH = AHH * 2 ^ (gamint * j);

    %-- get position of coefs
    lesx = 1 : 2 ^ j : sidata(2);
    lesy = 1 : 2 ^ j : sidata(1);

    [i1, j1] = find (isfinite (ALH));
    [i2, j2] = find (isfinite (AHL));
    [i3, j3] = find (isfinite (AHH));
    ii1 = max ([min(i1) min(i2) min(i3)]);
    ii2 = min ([max(i1) max(i2) max(i3)]);
    jj1 = max ([min(j1) min(j2) min(j3)]);
    jj2 = min ([max(j1) max(j2) max(j3)]);

    %-- store results
    coef.xpos{j} = lesx(jj1 :jj2);
    coef.ypos{j} = lesy(ii1 : ii2);
    coef.value{j}(:, :, 1) = ALH(ii1 : ii2, jj1 : jj2);  % x
    coef.value{j}(:, :, 2) = AHL(ii1 : ii2, jj1 : jj2);  % y
    coef.value{j}(:, :, 3) = AHH(ii1 : ii2, jj1 : jj2);  % xy

    sz = size (coef.value{j});
    coef.nj(j) = 3 * sz(1) * sz(2);

    %- Compute leaders sans_voisin for current scale
    if j<j1_ini
        sz = size (coef.value{j});
        leader.value{j} = NaN (sz(1 : 2));
        leader.xpos{j} = coef.xpos{j};
        leader.ypos{j} = coef.ypos{j};
        leader.nj(j) = prod (size (leader.value{j}));
    else
        if j==j1_ini
            sans_voisin = zeros ([size(ALH) 3]);
            sans_voisin(:, :, 1) = abs(ALH);  % x
            sans_voisin(:, :, 2) = abs(AHL);  % y
            sans_voisin(:, :, 3) = abs(AHH);  % xy
            if p ~= inf    % p-leaders
                sans_voisin = 2 ^ (2 * j) .* sans_voisin .^ p;
            end
        else
            sz = size (sans_voisin);
            nc = floor (sz(1 : 2) / 2);

            %-- get max at smaller scales
            sans_voisin_new = zeros ([nc 3]);
            for dir = 1 : 3
                sans_voisin_new(:, :, dir) = ...
                    compute_leader_sans_voisin (ALH, sans_voisin(:,:,dir), nc, p, j);
            end
            sans_voisin = sans_voisin_new;
        end

        %-- on prend le max sur les 8 voisins i.e. 9 coeffs
        lead_value = zeros (size (sans_voisin));
        for i = 1 : size (coef.value{j}, 3)
            lead_value(:, :, i) = ...
                compute_leader_from_neigbourhood (sans_voisin(:, :, i), p);
        end

        %-- get the position of leaders
        [i1, j1] = find (isfinite (lead_value(:, :, 1)));
        [i2, j2] = find (isfinite (lead_value(:, :, 2)));
        [i3, j3] = find (isfinite (lead_value(:, :, 3)));

        ii1 = max ([min(i1) min(i2) min(i3)]);
        ii2 = min ([max(i1) max(i2) max(i3)]);
        jj1 = max ([min(j1) min(j2) min(j3)]);
        jj2 = min ([max(j1) max(j2) max(j3)]);

        %-- FIXME: this only removes the border effects, and not the leaders that might be NaN because
        %- of missing data like in the 1d version.
        leader.xpos{j} = lesx(jj1 : jj2);
        leader.ypos{j} = lesy(ii1 : ii2);

        if p == inf
            leader.value{j} = max (lead_value(ii1 : ii2, jj1 : jj2, :), [], 3);
        else
            leader.value{j} = sum (lead_value(ii1 : ii2, jj1 : jj2, :), 3);
        end

        sz = size (leader.value{j});
        leader.nj(j) = sz(1) * sz(2);
    end
end % for j=1:nbvoies

%-- p-Leaders finish off
if p ~= inf
    for j=1:nbvoies
        leader.value{j} = (2 ^ (-2 * j) .* leader.value{j}) .^ (1 / p);
    end
end

%-- Pack estimation parameters:
coef.gamint     = gamint;
coef.Nwt       = Nwt;
coef.imagedata = true;

leader.gamint      = gamint;
leader.Nwt        = Nwt;
leader.p          = p;
leader.imagedata  = true;
%-- NLcorrPL will pack zp

%-- correct p-leaders for nonlinearity
[leader] = NLcorrPL2d(coef,leader,param_est);
coef.zp  = leader.zp;

end  % function DxPx2d
     %===============================================================================
     %- ANCILLARY SUBROUTINES
     %===============================================================================
    function [sansv] = compute_leader_sans_voisin(coef, sansv, nc, p, j)
    % Computes leaders without neighbours from coefs at current scale (j) and
    % leaders sans_voisin at the previous scale.
    % nc indicates the numer of coefficients at curr scale that will be used

        tmp(:, :, 1) = abs (coef(1 : nc(1), 1 : nc(2)));
        if p ~= inf    % p-leaders
            tmp(:, :, 1) = 2 ^ (2 * j) .* tmp(:, :, 1) .^ p;
        end
        tmp(:, :, 2) = sansv(1 : 2 : 2*nc(1) , 1 : 2 : 2*nc(2));
        tmp(:, :, 3) = sansv(2 : 2 : 2*nc(1) , 1 : 2 : 2*nc(2));
        tmp(:, :, 4) = sansv(1 : 2 : 2*nc(1) , 2 : 2 : 2*nc(2));
        tmp(:, :, 5) = sansv(2 : 2 : 2*nc(1) , 2 : 2 : 2*nc(2));

        if p == inf
            sansv = max (tmp, [], 3);  % Trailing singleton dim is squeezed out
        else
            sansv = sum (tmp, 3);  % Trailing singleton dim is squeezed out
        end

    end  % compute_leader_sans_voisin
         %-------------------------------------------------------------------------------
    function [leader] = compute_leader_from_neigbourhood (sansv, p)
    % Computes each leader from all leaders_sans_voisin in the neighbourhood

        si = size (sansv);
        ls = zeros (2 + si(1), 2 + si(2));
        ls(2 : end-1, 2 : end-1) = sansv;
        tmp(:, :, 1) = ls(1 : end - 2, 1 : end - 2);
        tmp(:, :, 2) = ls(1 : end - 2, 2 : end - 1);
        tmp(:, :, 3) = ls(1 : end - 2, 3 : end    );
        tmp(:, :, 4) = ls(2 : end - 1, 1 : end - 2);
        tmp(:, :, 5) = ls(2 : end - 1, 2 : end - 1);
        tmp(:, :, 6) = ls(2 : end - 1, 3 : end    );
        tmp(:, :, 7) = ls(3 : end    , 1 : end - 2);
        tmp(:, :, 8) = ls(3 : end    , 2 : end - 1);
        tmp(:, :, 9) = ls(3 : end    , 3 : end    );

        if p == inf    % wavelet leaders
            leader = max (tmp, [], 3);
        else    % p-leaders
            leader = sum (tmp, 3);
        end

    end
    %-------------------------------------------------------------------------------