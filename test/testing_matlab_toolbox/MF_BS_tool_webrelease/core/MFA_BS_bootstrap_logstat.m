function [logstatBS] = MFA_BS_bootstrap_logstat(coef, param_est, param_bs);
% function [logstatDyad, parBSest] = MFA_BS_bootstrap_logstat(data, method_mrq, param_est, param_bs);
% Compute bootstrap replications of  structure functions 
%
% Precodition: 1st level bootstrap is actually needed (i.e. n_resamp_1 and such params are valid)
%
% Herwig Wendt, Lyon, 2006 - 2008

%===============================================================================
% SETUP
%===============================================================================

doB2 = param_bs.n_resamp_2 > 0;
flag_need_B2 = any (param_bs.ci_method < 4 | param_bs.ci_method > 6);

j1 = param_est.j1;
if param_bs.flag_bs_range
    j1bs = j1;
else
    j1bs=1;
end

if doB2 && flag_need_B2
    TTsave=1;
else
    TTsave=0;
end

% Append fhandle to param_est (needed by resample functions)
% This is the function that will compute estimates from coefs.
%param_est.fhandle='flexEstFun_MFA';  
% This is already in param_est

nbvoies = length (coef.value);

%===============================================================================
% BOOSTRAP
%===============================================================================
if param_bs.bs_type  % TIME SCALE BOOTSTRAP
    %-- Compute absolute value of coefficients before calling!
    coef.value = cellfun (@abs, coef.value, 'UniformOutput', false);
    
    %-- Compute bootstrap resamples
    if ~param_est.imagedata
        [estimates] = resample1d_T_S (coef, param_bs, param_est);
    else
        [estimates] = resample2d_T_S (coef, param_bs, param_est);
    end
    
    %-- Convert output
    for j = j1bs : nbvoies
        estB(:, :, j) = estimates{j}.T;
        Vest(:, j)    = estimates{j}.stdt;
        if doB2 
            estBB(:, :, :, j) = estimates{j}.TT;
        end
    end
else   % ORDINARY BOOTSTRAP
    %-- Checl block size
    blen     = param_bs.block_size; 
    blen_vec = blen * ones(1, nbvoies);
    for j = 1 : nbvoies
        while (coef.nj(j) < 2 * blen) && (blen ~= 1) % check if block size is appropriate
            blen = max (fix (blen / 2), 1);
            blen_vec(j : nbvoies) = blen * ones (size (j : nbvoies));
        end
        param_bs.block_size = blen;
       
        if j < j1bs    % don't do bootstrap at this scale
            continue
        end

        %-- Compute boostrap resamples at this scale
        if ~param_est.imagedata
            [estimates] = resample1d (abs (coef.value{j}), param_bs, param_est, TTsave);
        else
            [estimates] = resample2d (abs (coef.value{j}), param_bs, param_est, TTsave);
        end
        
        %-- Convert output
        estB(:, :, j) = estimates.T;
        Vest(:, j)    = estimates.stdt;
        if doB2
            estBB(:, :, :, j) = estimates.TT;
        end
    end  % for j=1:nbvoies
end

%===============================================================================
% Output
%===============================================================================
logstatBS.estB = estB;
logstatBS.Vest = Vest;
if doB2
    logstatBS.estBB = estBB;
end
