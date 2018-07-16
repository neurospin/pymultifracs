function my_click_and_plot(h)
%
% Stephane G. Roux, Lyon, 2005

newfig = figure;
newaxes = copyobj(h,newfig);
set(newaxes,'units','normalized','position',[0.13 0.11 0.775 0.815]);
%set(newaxes,'ButtonDownFcn','');
