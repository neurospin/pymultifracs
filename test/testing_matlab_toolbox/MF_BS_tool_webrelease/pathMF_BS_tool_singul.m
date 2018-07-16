%%%%%% SETTING THE PATHS FOR WLBMF_tool
%
% Herwig Wendt, Lyon, 2006 - 2008

tmp=which('pathMF_BS_tool_singul'); index=strfind(tmp,filesep);
p=tmp(1:index(end));

addpath([p,'core'],[p,'misc']);

clear p tmp index
