function plot_logscale (ax, plot_data, param_est, param_bs)
% 
%   - plot_data: structure with stuff to plot and parameters. Fields
%      * yy: "structure function"
%      * yy_bs: bootstrap replicates of "structure functions"
%      * slope, inter: slope and intercept of estimated linear model
%      * ylabel, title

if ~isfield (plot_data, 'fontsize_small')
    fontsize_small = 8;
else
    fontsize_small = plot_data.fontsize_small;
end

if ~isfield (plot_data, 'fontsize_large')
    fontsize_large = 12;
else
    fontsize_large = plot_data.fontsize_large;
end    

jj = 1 : length (plot_data.yy);  % x data
yy = plot_data.yy;  % y data
if isfield (plot_data, 'yy_bs') && ~isempty (plot_data.yy_bs)
    doBS = true;
else
    doBS = false;
end

scal_rng = param_est.j1 : param_est.j2;   % Indices of scales used for bootstrap
linear_est = plot_data.slope * jj + plot_data.inter;

axes (ax);
hold on
%-- 1: plot confidence intervals
if doBS
    [ci_lo, ci_hi] = get_bs_ci_limits (plot_data.yy_bs, param_bs);
    jj_bs = plot_data.idx_bs;
    errorbar (jj(jj_bs), yy(jj_bs), abs(ci_lo(jj_bs)' - yy(jj_bs)), abs(ci_hi(jj_bs)' - yy(jj_bs)), 'r');
end
%-- 2: plot estimates
plot (jj, yy, 'k.-')

%-- 3:  plot linear estimate in full range
plot (jj, linear_est, 'b--')

%-- 4: emphasize linear estimate in used scaling range
plot (jj(scal_rng), linear_est(scal_rng), 'b-')
hold off
grid on
if plot_data.slope < 0
    posX = 0.7;
else
    posX = 0.03;
end
text ('String', num2str(plot_data.slope), ...
      'FontSize', fontsize_large, ...
      'FontName', 'Helvetica', ...
      'Color', 'red', ...
      'Units','normalized', ...
      'Pos',[posX 0.9]);
text ('String', sprintf ('wtype %i', param_est.wtype), ...
      'FontSize', fontsize_small, ...
      'FontName', 'Helvetica', ...
      'Units', 'normalized', ...
      'Pos',[posX 0.76]);
text ('String', sprintf ('j:[%i-%i]', param_est.j1, param_est.j2), ...
      'FontSize', fontsize_small, ...
      'FontName', 'Helvetica', ...
      'Units', 'normalized', ...
      'Pos',[posX 0.69]);

% Setup callback for plotting in separate figure if clicked:
children = get (ax, 'Children');
set (children, 'HitTest', 'off');
set (ax, 'ButtonDownFcn', 'my_click_and_plot (gcbo)')