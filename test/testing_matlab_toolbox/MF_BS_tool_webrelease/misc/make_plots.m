function make_plots (logstat, est, fig_num)

% Makes the plots of all results.

%===============================================================================
%--- SETUP
%===============================================================================

% --- Constants ---
loge = log2 (exp (1));

% --- Determine which methods were used ----
flag_DWT = isfield (logstat, 'DWT');
flag_LWT = isfield (logstat, 'LWT');

if ~flag_DWT | ~flag_LWT
    error ('Either DWT or LWT must be computed')
end

if isfield (logstat.DWT, 'param_est') 
    param_est = logstat.DWT.param_est;
    imagedata = logstat.DWT.imagedata;
else
    param_est = logstat.LWT.param_est;
    imagedata = logstat.LWT.imagedata;
end

if isfield (logstat.DWT, 'BS') 
    param_bs = logstat.DWT.param_bs;
elseif isfield (logstat.LWT, 'BS')
    param_bs = logstat.LWT.param_bs;
else
    param_bs = [];
end

if ~isempty (param_bs)
    doBS = param_bs.n_resamp_1 > 0;
else
    doBS = false;
end

% --- Determine what estimates where computed ---
[ZQ, DH, CP] = which_estimates (param_est.estimate_select);
[idx_zq, idx_dq, idx_hq idx_cum] = get_estimate_id (param_est);


% --- Setup figure flags ---
if (ZQ|CP)
    PlotLogScale=1;
else 
    PlotLogScale=0; 
end

PlotResult=1;
figs=PlotLogScale+PlotResult;


% --- Method dependent strings for titles ---
meth_name = [];
meth_str = [];
if flag_DWT 
    meth_name = [meth_name {'DWT'}];
    meth_str = [meth_str {'DWT'}];
end
if flag_LWT
    meth_name = [meth_name {'LWT'}];
    meth_str = [meth_str {sprintf('PWT p=%g', param_est.p)}];
end
num_meth = length (meth_name);

switch param_est.type_exponent,
  case 0
    expstr=['hp'];
    parstr=[''];
  case 1
    expstr=['Oscillation'];
    parstr=['d\gamma=',num2str(param_est.delta_gam)];
  case 2
    expstr=['Lacunary'];
    parstr=['d 1/p=',num2str(param_est.delta_p)];
  case 3
    expstr=['Cancellation'];
    parstr=['d\gamma=',num2str(param_est.delta_gam)];
end

%-------------------------------------------------------------------------------
% BOXPLOTS OF CUMULANTS
%-------------------------------------------------------------------------------
if PlotResult && CP
    figure (fig_num); clf
    for im = 1 : num_meth  % loop methods
        for ic = 1 : param_est.cum
            subplot (param_est.cum, num_meth, im + (ic-1) * num_meth)
            
            plot_data.yy = est.(meth_name{im}).t(idx_cum(ic));
            if doBS
                plot_data.yy_bs = est.(meth_name{im}).BS.T(:, idx_cum(ic));
            else
                plot_data.yy_bs = [];
            end
            
            plot_boxplot (gca, plot_data, param_est, param_bs);
            title (sprintf ('%s c_%i %s', meth_str{im}, ic,  expstr));
        end
    end
end

%-------------------------------------------------------------------------------
% SCALING FUNCTION AND MULTIFRACTAL SPECTRUM
%-------------------------------------------------------------------------------
if PlotResult && (DH | ZQ)
    figure (fig_num + 1); clf
    for im = 1 : num_meth  % loop methods
        %-----------------------------------
        %- zeta(q)
        if ZQ
            subplot (2, num_meth, im)
            
            plot_data.xx = param_est.q;
            plot_data.yy = est.(meth_name{im}).t(idx_zq);
            if doBS
                plot_data.xx_bs = [];
                plot_data.yy_bs = est.(meth_name{im}).BS.T(:, idx_zq);
            end
            plot_data.p0noint = est.(meth_name{im}).p0noint;
            plot_data.p0 = est.(meth_name{im}).p0;

            plot_xy (gca, plot_data, param_est, param_bs);
            xlabel ('q');
            ylabel ('\zeta(q)');
            title (sprintf ('%s, \\zeta(q), %s', meth_str{im}, expstr))
        end
        %-----------------------------------
        %- D(h)
        if DH
            subplot (2, num_meth, im + num_meth)
            
            plot_data.xx = est.(meth_name{im}).t(idx_hq);
            plot_data.yy = est.(meth_name{im}).t(idx_dq);
            if doBS
                plot_data.xx_bs = est.(meth_name{im}).BS.T(:, idx_hq);
                plot_data.yy_bs = est.(meth_name{im}).BS.T(:, idx_dq);
            end
            plot_data.p0noint = est.(meth_name{im}).p0noint;
            plot_data.p0 = est.(meth_name{im}).p0;
            
            plot_xy (gca, plot_data, param_est, param_bs);
            xl = xlim;
            xl(1) = min (xl(1), 0);
            xl(2) = max (xl(2), 1.5);
            xlim (xl)
            yl = ylim;
            yl(1) = 0;
            yl(2) = max (yl(2), (imagedata + 1) * 1.05);
            ylim (yl)
            xlabel ('h');
            ylabel ('D(h)');
            title (sprintf ('%s, D(h), %s', meth_str{im}, expstr))
        end            
    end
end

%-------------------------------------------------------------------------------
% LOGSCALE DIAGRAMS: CUMULANTS
%-------------------------------------------------------------------------------
if PlotLogScale && CP
    figure (fig_num + 2); clf
    for im = 1 : num_meth  % loop methods
        for ic = 1 : param_est.cum
            subplot (param_est.cum, num_meth, im + (ic-1) * num_meth)
            
            plot_data.yy        = logstat.(meth_name{im}).est(idx_cum(ic), :) * loge;
            if doBS
                plot_data.yy_bs = logstat.(meth_name{im}).BS.estB(:, idx_cum(ic), :) * loge;
                plot_data.idx_bs = get_bs_used_j ();
            else
                plot_data.yy_bs = [];
                plot_data.idx_bs = [];
            end
            plot_data.slope     = est.(meth_name{im}).t(idx_cum(ic));
            plot_data.inter     = est.(meth_name{im}).aest(idx_cum(ic));
            
            plot_logscale (gca, plot_data, param_est, param_bs);
            ylabel (sprintf ('C_{%i}(j)', ic));
            title (sprintf ('%s, c_%i, LogScale, %s', meth_str{im}, ic,  expstr));
        end
    end  % for im = 1 : length (meth_name) 
end % if PlotLogScale && CP

%-------------------------------------------------------------------------------
% LOGSCALE DIAGRAMS: STRUCTURE FUNCTIONS
%-------------------------------------------------------------------------------
if PlotLogScale && ZQ
    num_q = length (param_est.q);
    
    % Build ordering for subplots
    div = ceil (num_q / 3);
    sbid = zeros (1, 3 * div); 
    sbid(1           : div    ) = 1 : 3 : 3 * div;
    sbid(div + 1     : 2 * div) = 2 : 3 : 3 * div;
    sbid(2 * div + 1 : end    ) = 3 : 3 : 3 * div;
    
    for im = 1 : num_meth  % loop methods
        figure (fig_num + 2 + im); clf
        for iq = 1 : num_q
            subplot (div, 3, sbid(iq))
            plot_data.yy        = logstat.(meth_name{im}).est(idx_zq(iq), :);
            
            if doBS
                plot_data.yy_bs = logstat.(meth_name{im}).BS.estB(:, idx_zq(iq), :);
                plot_data.idx_bs = get_bs_used_j ();
            else
                plot_data.yy_bs = [];
                plot_data.idx_bs = [];
            end
            plot_data.slope     = est.(meth_name{im}).t(idx_zq(iq));
            plot_data.inter     = est.(meth_name{im}).aest(idx_zq(iq));
            
            plot_logscale (gca, plot_data, param_est, param_bs);
            ylabel ('log_2 S(q, j)');
            title (sprintf ('%s, \\zeta(q), LogScale, %s, q = %g', ...
                            meth_str{im},  expstr, param_est.q(iq)));
        end
    end
end

%-------------------------------------------------------------------------------
% LOGSCALE DIAGRAMS: SUPCOEF, MINCOEF
%-------------------------------------------------------------------------------
if PlotLogScale
    figure (fig_num + 5); clf
    
    %-----------------------------------
    %- h_min, coefs
    subplot (1, 3, 1)
    
    plot_data.yy     = log2 (logstat.DWT.supcoef);
    plot_data.yy_bs  = [];
    plot_data.slope  = est.DWT.h_min;
    plot_data.inter  = est.DWT.h_min_aest;
    
    plot_logscale (gca, plot_data, param_est, param_bs);
    ylabel ('log_2 sup [d(j,.)]');
    title ('h_{min}, Coeff, LogScale');

    %-----------------------------------
    %- h_min, leaders
    subplot (1, 3, 2)
    
    plot_data.yy     = log2 (logstat.LWT.supcoef);
    plot_data.yy_bs  = [];
    plot_data.slope  = est.LWT.h_min;
    plot_data.inter  = est.LWT.h_min_aest;
    
    plot_logscale (gca, plot_data, param_est, param_bs);
    ylabel ('log_2 sup [L(j,.)]');
    title ('h_{min}, Leaders, LogScale');

    %-----------------------------------
    %- h_max, leaders
    subplot (1, 3, 3)
    
    plot_data.yy     = log2 (logstat.LWT.mincoef);
    plot_data.yy_bs  = [];
    plot_data.slope  = est.LWT.h_max;
    plot_data.inter  = est.LWT.h_max_aest;
    
    plot_logscale (gca, plot_data, param_est, param_bs);
    ylabel ('log_2 inf [L(j,.)]');
    title ('h_{max}, Leaders, LogScale');
end


function [idx] = get_bs_used_j ()
    if logstat.(meth_name{im}).param_bs.flag_bs_range
        % Bootstrap done only in scaling range
        jmin = logstat.(meth_name{im}).param_est.j1;
        jmax = logstat.(meth_name{im}).param_est.j2;
    else
        % Bootstrap done in all scale
        jmin = 1;
        jmax = length (logstat.(meth_name{im}).nj);
    end
    [idx] = jmin : jmax;
end

end  % function make_plots
