function [varargout] = MF_BS_tool(data, param_est, method_mrq, param_bs, param_test, verbosity, fig_num);
% function [varargout] = MFA_BS_tool(data, param_est, method_mrq, param_bs, param_test, verbosity, fig_num);
% - Calculate cp, (zeta(q), D(h))
% - Bootstrap CI
% - Bootstrap tests
% Analysis Methods: DWT, LWT
%
% SEE demo_MF_BS_tool for usage, outputs and parameters
%
% Herwig Wendt, Lyon, 2006 - 2008
%
%*******************************************************************************
% If you use the toolbox, please quote:
%
%     @ARTICLE{WendtSPM2007,
%       author = {Herwig Wendt and Patrice Abry and St??phane Jaffard},
%       title = {Bootstrap for Empirical Multifractal Analysis},
%       journal = {IEEE Signal Proc. Mag.},
%       volume = {24},
%       number = {4},
%       pages = {38--48},
%       year = {2007},
%     }
%
% and (or)
%
%     @ARTICLE{WendtSP2009,
%       author = {Herwig Wendt and St??phane G. Roux and Patrice Abry and St??phane Jaffard},
%       title = {Wavelet leaders and bootstrap for multifractal analysis of images},
%       journal = {Signal Proces.},
%       volume = {89},
%       pages = {1100--1114},
%       year = {2009},
%     }
%
%*******************************************************************************

%===============================================================================
% SETUP
%===============================================================================

if isempty(param_bs.ci_method) | (param_bs.n_resamp_1<=1)
    doBS = 0;
else
    doBS=1;
end

%-- determine which analysis are to be done
if length (find (method_mrq == 1))
    do_DWT=1;
else
    do_DWT=0;
end
if length (find (method_mrq == 2))
    do_LWT=1;
else
    do_LWT=0;
end

if isstruct(data) % data is structure functions or multiresolution quantities

    if isfield (data, 'DWT') || isfield (data, 'LWT')  % if structure function
        flag_compute_logstat = 0;
        logstat = data;
        clear data

        %-- determine if required analysis can be done
        if do_DWT && ~isfield (logstat, 'DWT')
            do_DWT = 0;
        end
        if do_LWT && ~isfield (logstat, 'LWT')
            do_LWT = 0;
        end

        %-- use stored param_est but with new scaling range
        j1=param_est.j1;
        j2=param_est.j2;
        if do_DWT
            param_est = logstat.DWT.param_est;
            j2max = size (logstat.DWT.est, 2);
        elseif do_LWT
            param_est = logstat.LWT.param_est;
            j2max = size (logstat.LWT.est, 2);
        else
        end

        param_est.j1 = j1;
        param_est.j2 = min (j2, j2max);

        % Adjust correction in case scaling range changed:
        if do_LWT
            logstat.LWT = adjust_correction (logstat.LWT, param_est.j1, param_est.j2);
        end

        logstat.DWT.param_est = param_est;
        logstat.LWT.param_est = param_est;
    elseif isfield (data, 'coef') || isfield (data, 'leader')
        % TODO here we should also readjust the correction term if the scaling range has changed.

        flag_compute_logstat = 1;
        flag_compute_mrq = 0;
        mrq = data;
        clear data

        %-- determine if required analysis can be done
        if do_DWT && ~isfield (mrq, 'coef')
            do_DWT = 0;
        end
        if do_LWT && ~isfield (mrq, 'leader')
            do_LWT = 0;
        end

        %-- use stored parameters
        if do_LWT
            param_est.gamint = mrq.leader.gamint;
            param_est.NWt    = mrq.leader.Nwt;
            param_est.p      = mrq.leader.p;
        elseif do_DWT
            param_est.gamint = mrq.coef.gamint;
            param_est.NWt    = mrq.coef.Nwt;
        else
        end
    end

else  % data is signal to be analyzed
    flag_compute_logstat = 1;
    flag_compute_mrq = 1;
end  % if isstruct (data)


% Check for scaling range selection:
if numel (param_est.j1) == 2 || numel (param_est.j2) == 2
    flag_select_scaling_range = true;
    jlimit = [param_est.j1 ; param_est.j2];
    param_est.j1 = 1;
    param_est.j2 = inf;
else
    flag_select_scaling_range = false;
end


% parameters for p0 estimation
param_est.P0est.q0 = 1e-2;
param_est.P0est.q1 = 1e2;
param_est.P0est.NR = 10;

%===============================================================================
%  GET LOGSCALE DIAGRAMS
%===============================================================================
if flag_compute_logstat
    if flag_compute_mrq
        %-- check size of data
        imagedata = ~isvector (data);
        param_est.imagedata = imagedata;

        if verbosity > 0; fprintf (' Computing ... \n'); end;

        %-- Compute coefficients and leaders if p > 0
        if param_est.p > 0
            if ~imagedata
                [mrq.coef, mrq.leader] = DxPx1d (data, param_est);
            else
                [mrq.coef, mrq.leader] = DxPx2d (data, param_est);
            end
            %-- Modify leaders if "fancy" exponents are required
            if param_est.type_exponent > 0
                [mrq.leader] = singulPx (data, mrq.leader, param_est);
            end
        elseif param_est.p == -1
            if imagedata
                error ('DFA is not supported for images.')
            end
            mrq.coef = TxDFA_1d (data, param_est.nwt, -param_est.p);
            mrq.leader = mrq.coef;
        elseif param_est.p == -2
            if imagedata
                error ('DFA is not supported for images.')
            end
            mrq.coef = TxMFDFA_1d (data, param_est.nwt, -param_est.p);
            mrq.leader = mrq.coef;
        else
            error ('Unsupported value for p')
        end
        mrq.coef.imagedata = imagedata;
        mrq.leader.imagedata = imagedata;

    end  % if flag_compute_mrq

    if do_DWT
        logstat.DWT = MFA_BS_compute_logstat (mrq.coef, param_est);
        logstat.DWT.imagedata = mrq.coef.imagedata;

        %- Update param_est (j2 might have been truncated):
        param_est = logstat.DWT.param_est;
    end

    if do_LWT
        logstat.LWT = MFA_BS_compute_logstat (mrq.leader, param_est);
        logstat.LWT.imagedata = mrq.leader.imagedata;
        try logstat.LWT.log_Spj = logstat.DWT.log_Spj; end

        %-- Update again (in case previous conditional was not executed)
        %- param_est should be equal in both cases
        param_est = logstat.LWT.param_est;
    end
else
    mrq = [];
end  % if flag_compute_logstat

%===============================================================================
% BOOTSTRAP LOGSTAT
%===============================================================================
if doBS
    if do_DWT
        logstat.DWT.BS = MFA_BS_bootstrap_logstat (mrq.coef, param_est, param_bs);
        logstat.DWT.param_bs = param_bs;
    end
    if do_LWT
        logstat.LWT.BS = MFA_BS_bootstrap_logstat (mrq.leader, param_est, param_bs);
        logstat.LWT.param_bs = param_bs;
    end
end

%===============================================================================
% DETERMINE SCALING RANGE IF NEEDED
%===============================================================================

% if j1 or j2 are 2d vectors, interpret them as search intervals
if flag_select_scaling_range && doBS
    zid = get_estimate_id (param_est);
    if do_LWT
        [j1o, j2o] = select_scaling_range (logstat.LWT, jlimit, zid, ...
                                         param_est.wtype);
        logstat.LWT.param_est.j1 = j1o;
        logstat.LWT.param_est.j2 = j2o;
        if do_DWT
            logstat.DWT.param_est.j1 = j1o;
            logstat.DWT.param_est.j2 = j2o;
        end
    else
        [j1o, j2o] = select_scaling_range (logstat.DWT, jlimit, zid, ...
                                         param_est.wtype)
        logstat.DWT.param_est.j1 = j1o;
        logstat.DWT.param_est.j2 = j2o;
    end
    param_est.j1 = j1o;
    param_est.j2 = j2o;
end

%===============================================================================
% GET FINAL ESTIMATES: LINEAR REGRESSIONS
%===============================================================================
if do_DWT
    [est.DWT] = MFA_BS_regrest (logstat.DWT, param_est);
end
if do_LWT
    [est.LWT] = MFA_BS_regrest (logstat.LWT, param_est);
end

%===============================================================================
% BOOTSTRAP ESTIMATES, CONFIDENCE INTERVALS, TESTS
%===============================================================================
stat = [];

if doBS
    if do_DWT
        est.DWT.BS = MFA_BS_bootstrap_regrest (logstat.DWT, param_est, param_bs);
        est.DWT.param_bs = param_bs;

        if param_bs.CI || param_bs.TEST
            stat.DWT = MFA_BS_compute_ci_test (est.DWT, param_bs, param_test);
        end
    end

    if do_LWT
        logstat.LWT.BS = MFA_BS_bootstrap_logstat (mrq.leader, param_est, param_bs);
        logstat.LWT.param_bs = param_bs;

        est.LWT.BS = MFA_BS_bootstrap_regrest (logstat.LWT, param_est, param_bs);
        est.LWT.param_bs = param_bs;

        if param_bs.CI || param_bs.TEST
            stat.LWT = MFA_BS_compute_ci_test (est.LWT, param_bs, param_test);
        end
    end
end



%===============================================================================
% OUTPUT
%===============================================================================
if nargout >= 1
    varargout{1} = est;
end
if nargout >= 2
    varargout{2} = logstat;
end
if nargout >= 3
    varargout{3} = stat;
end
if nargout >= 4
    varargout{4} = mrq;
end


%===============================================================================
% DISPLAY
%===============================================================================

if mod (verbosity, 10) > 0
    make_plots (logstat, est, fig_num);
end

if mod (floor (verbosity / 10), 10) > 0
    display_tables (logstat, est, stat);
end
