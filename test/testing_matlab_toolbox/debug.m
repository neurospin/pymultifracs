clear all
close all

load('../test_data/mrw07005n32768.mat')

% Initialize MF object with global parameters
mf_obj = MF_BS_tool_inter;
mf_obj.method_mrq = [1 2];
mf_obj.cum     = 3;
mf_obj.verbosity =0;

% set parameters and analyze data
mf_obj.gamint = 0;
mf_obj.j1     = 3;
mf_obj.j2     = 12;
mf_obj.nwt    = 3;
mf_obj.q      = linspace(-10, 10, 50);%-8:8;
mf_obj.wtype  = 0;
mf_obj.p      = inf;

mf_obj.analyze (data);

% Get results
cid = mf_obj.get_cid ();  % Indices of c_p
zid = mf_obj.get_zid ();  % Indices of zeta(q)
hid = mf_obj.get_hid ();  % Indices of h(q)
Did = mf_obj.get_Did ();  % Indices of D(q)

cp = mf_obj.est.LWT.t(cid);  % Estimates of c_p
zq = mf_obj.est.LWT.t(zid);  % Estimates of zeta(q)
hmin = mf_obj.est.DWT.h_min;  % estimate of hmin
eta_p = mf_obj.est.LWT.zp;    % estimate of eta_p

hq = mf_obj.est.LWT.t(hid);  % Estimates of h(q)
Dq = mf_obj.est.LWT.t(Did);  % Estimates of D(q)
