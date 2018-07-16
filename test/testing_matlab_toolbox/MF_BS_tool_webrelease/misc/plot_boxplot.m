function plot_boxplot (ax, plot_data, param_est, param_bs)
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

if ~isfield (plot_data, 'box_width')
    box_width = 0.5;
else
    box_width = plot_data.box_width;
end    


if isfield (plot_data, 'yy_bs') && ~isempty (plot_data.yy_bs)
    doBS = true;
else
    doBS = false;
end


axes (ax);
hold on
% 1: Plot boxplot
if doBS
    boxplot (plot_data.yy_bs(:), 'notch', 'on', 'widths', box_width);
end

% 2: superimpose estimates
plot ([0.5 1 - box_width / 2 - 0.05], plot_data.yy * [1 1], 'k')
plot ([0.5 1 + box_width / 2 + 0.05], plot_data.yy * [1 1], 'k')

hold off
grid on
set (gca, 'XTickLabel', ' ');
text ('String', num2str (plot_data.yy), ...
      'FontSize', fontsize_large, ...
      'FontName', 'Helvetica', ...
      'Color', 'red', ...
      'Units','normalized', ...
      'Pos',[0.03 0.9]);
text ('String', sprintf ('Frac Int %g', param_est.gamint), ...
      'FontSize', fontsize_small, ...
      'FontName', 'Helvetica', ...
      'Units', 'normalized', ...
      'Pos', [0.03 0.83]);
text ('String', sprintf ('N_{\\psi} %i', param_est.nwt), ...
      'FontSize', fontsize_small, ...
      'FontName', 'Helvetica', ...
      'Units', 'normalized', ...
      'Pos', [0.03 0.76]);
text ('String', sprintf ('j:[%i-%i]', param_est.j1, param_est.j2), ...
      'FontSize', fontsize_small, ...
      'FontName', 'Helvetica', ...
      'Units', 'normalized', ...
      'Pos', [0.03 0.69]);
text ('String', sprintf ('wtype %i', param_est.wtype), ...
      'FontSize', fontsize_small, ...
      'FontName', 'Helvetica', ...
      'Units','normalized', ...
      'Pos', [0.03 0.62]);

% Setup callback for plotting in separate figure if clicked:
children = get (ax, 'Children');
set (children, 'HitTest', 'off');
set (ax, 'ButtonDownFcn', 'my_click_and_plot (gcbo)')