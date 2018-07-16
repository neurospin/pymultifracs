function [param_est, param_bs, param_test]=MF_BS_tool_param_singul(nwt, ...
                                                  gamint,delta_gam,delta_p,type_exponent,j1,j2,wtype,estimate_select,cum,q,n_resamp_1,n_resamp_2,block_size,ci_method,alpha,flag_bs_range,null_type,null_param,bs_type,CI,TEST,p,j1_ini,sym, plead_corr);
% function [param_est, param_bs, param_test]=MF_BS_tool_param_singul(nwt,gamint,delta_gam,delta_p,type_exponent,j1,j2,wtype,estimate_select,cum,q,n_resamp_1,n_resamp_2,block_size,ci_method,alpha,flag_bs_range,null_type,null_param,bs_type,CI,TEST,p,j1_ini,sym);
%
% Herwig Wendt, TLS, 01/2015

% ---> RFL, 2015-02-23 --->
param_est.plead_corr = plead_corr;
%if p == 0; plead_corr = 0; end % no nonlinearity for standard leaders
% if p == 0; plead_corr = 0; else; plead_corr = 1; end % no nonlinearity for standard leaders
% <--- RFL, 2015-02-23 <---

% if plead_corr==1 % ensure first cumulant to be estimated
%     if ~rem(bin2dec(num2str(estimate_select)),2); estimate_select = estimate_select + 1; cum = 1; end
% end
if j1_ini>j1 % check that all scales available for estimation
    j1_ini=j1; disp(['Set j1_ini=',num2str(j1_ini),' (selected j1)']);
end
%%%% Estimation
param_est.nwt = nwt;
param_est.sym=sym;
param_est.j1_ini=j1_ini;
param_est.gamint=gamint;
param_est.delta_gam = delta_gam;
param_est.delta_p = delta_p;
param_est.type_exponent = type_exponent;
param_est.j1=j1;
param_est.j2=j2;
param_est.wtype=wtype;
param_est.estimate_select = estimate_select; % 0: only cp; 1: zeta(q), D(h), cp
param_est.cum = cum;
param_est.q=q;
param_est.p = p;
%%%% Bootstrap
param_bs.n_resamp_1=n_resamp_1;
param_bs.n_resamp_2=n_resamp_2;
param_bs.blocklength=block_size;
param_bs.ci_method=ci_method;
param_bs.alpha=alpha;
param_bs.flag_bs_range=flag_bs_range;
param_bs.bs_type = bs_type;
param_bs.CI=CI;
param_bs.TEST=TEST;
param_test.null_type=null_type;
param_test.null_param=null_param;
