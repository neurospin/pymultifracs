function plot_xy (ax, plot_data, param_est, param_bs)
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

% Unpack data
xx = plot_data.xx;  % x data
yy = plot_data.yy;  % y data
if isfield (plot_data, 'xx_bs') && ~isempty (plot_data.xx_bs) 
    doBS_x = true;
    xx_bs = plot_data.xx_bs;
else
    doBS_x = false;
end
if isfield (plot_data, 'yy_bs') && ~isempty (plot_data.yy_bs) 
    doBS_y = true;
    yy_bs = plot_data.yy_bs;
else
    doBS_y = false;
end

axes (ax);
hold on
%-- 1: plot confidence intervals, x
if doBS_x
    [ci_lo, ci_hi] = get_bs_ci_limits (xx_bs, param_bs);
    for iq = 1 : length (xx)
        plot ([ci_lo(iq) ci_hi(iq)], yy(iq) * [1 1], 'r')
    end
end

%-- 2: plot confidence intervals, y
if doBS_y
    [ci_lo, ci_hi] = get_bs_ci_limits (yy_bs, param_bs);
    for iq = 1 : length (xx)
        plot (xx(iq) * [1 1], [ci_lo(iq) ci_hi(iq)], 'r')
    end
end

%-- 3: plot estimates
plot (xx, yy, 'k.-')
hold off
grid on

text ('String', sprintf ('Frac Int %g', param_est.gamint), ...
      'FontSize', fontsize_small, ...
      'FontName', 'Helvetica', ...
      'Units', 'normalized', ...
      'Pos', [0.03 0.93]);
text ('String', sprintf ('N_{\\psi} %i', param_est.nwt), ...
      'FontSize', fontsize_small, ...
      'FontName', 'Helvetica', ...
      'Units', 'normalized', ...
      'Pos', [0.03 0.86]);
text ('String', sprintf ('j:[%i-%i]', param_est.j1, param_est.j2), ...
      'FontSize', fontsize_small, ...
      'FontName', 'Helvetica', ...
      'Units', 'normalized', ...
      'Pos', [0.03 0.79]);
text ('String', sprintf ('wtype %i', param_est.wtype), ...
      'FontSize', fontsize_small, ...
      'FontName', 'Helvetica', ...
      'Units','normalized', ...
      'Pos', [0.03 0.72]);
text ('String', sprintf ('(p_0=%i)', round (plot_data.p0noint * 100) / 100), ...
      'FontSize', fontsize_large, ...
      'FontName', 'Helvetica', ...
      'Color', 'red', ...
      'Units', 'normalized', ...
      'pos', [0.5 0.2], ...
      'HorizontalAlignment', 'center');
text ('String', sprintf ('p_0^{(\\gamma)}=%i', round (plot_data.p0 * 100) / 100), ...
      'FontSize', fontsize_large, ...
      'FontName', 'Helvetica', ...
      'Color', 'red', ...
      'Units', 'normalized', ...
      'Pos', [0.5 0.07], ...
      'HorizontalAlignment', 'center');

% Setup callback for plotting in separate figure if clicked:
children = get (ax, 'Children');
set (children, 'HitTest', 'off');
set (ax, 'ButtonDownFcn', 'my_click_and_plot (gcbo)')