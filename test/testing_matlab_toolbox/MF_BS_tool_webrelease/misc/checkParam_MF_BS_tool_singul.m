% checkParam_MF_BS_tool.m
%
% Check if all necessary parameters for 
% MFA_BS_light_param.m are defined
% In case of, give them default values
%
% Herwig Wendt, Lyon, 2006 - 2008

try 
    CI;
catch;
    CI=0; end
try 
    TEST;
catch;
    TEST=0; end
try 
    null_type;
catch;
    null_type=0; TEST=0; end
try 
    null_param;
catch;
    null_param=0; TEST=0; end
try 
    n_resamp_1; 
catch; 
    n_resamp_1=49; disp(['Set n_resamp_1=',num2str(n_resamp_1),' (default)']); end;
try 
    n_resamp_2; 
catch; 
    n_resamp_2=49; disp(['Set n_resamp_2=',num2str(n_resamp_2),' (default)']); end; 
try 
    method_mrq; 
catch; 
    method_mrq = [2]; disp(['Set method_mrq = ',num2str(method_mrq),' (default)']);  end;
try 
    bs_type; 
catch; 
    bs_type = 0; disp(['Set bs_type = ',num2str(bs_type),' (default)']);  end;
try 
    nwt; 
catch; 
    nwt = 3; disp(['Set nwt = ',num2str(nwt),' (default)']);  end;
try 
    gamint; 
catch; 
    gamint=0; disp(['Set gamint=',num2str(gamint),' (default)']);  end;
try 
    verbosity; 
catch; 
    verbosity=2; disp(['Set verbosity=',num2str(verbosity),' (default)']);  end;
try 
    fig_num; 
catch; 
    fig_num = 0; disp(['Set fig_num = ',num2str(fig_num),' (default)']);  end;
try 
    alpha; 
catch; 
    alpha=0.1; disp(['Set alpha=',num2str(alpha),' (default)']);  end;
try 
    ci_method; 
catch; 
    ci_method=3; disp(['Set ci_method=',num2str(ci_method),' (default)']);  end;
try 
    q; 
catch; 
    q=2; disp(['Set q=',num2str(q),' (default)']);  end;
try 
    cum; 
catch; 
    cum = 2; disp(['Set cum = ',num2str(cum),' (default)']);  end;
try 
    flag_bs_range; 
catch; 
    flag_bs_range=1; disp(['Set flag_bs_range=',num2str(flag_bs_range),' (default)']);  end;
try 
    estimate_select; 
catch; 
    estimate_select = 101; disp(['Set estimate_select = ',num2str(estimate_select),' (default)']);  end;
try 
    wtype; 
catch; 
    wtype=1; disp(['Set wtype=',num2str(wtype),' (default)']);  end;
try 
    j1; 
catch; 
    j1=3; disp(['Set j1=',num2str(j1),' (default)']);  end;
try 
    sym; if sym~=0&sym~=1; sym=1; disp(['Set sym=',num2str(sym)]); end
catch; 
    sym=0; disp(['Set sym=',num2str(sym),' (default)']);  end;
try 
    j1_ini; if j1_ini<1; j1_ini=1; disp(['Set j1_ini=',num2str(j1_ini)]); end
catch; 
    j1_ini=1; disp(['Set j1_ini=',num2str(j1_ini),' (default)']);  end
try 
    p; 
catch; 
    p = 2; disp(['Set p = ',num2str(p),' (default)']);  end;
if ~bs_type;
    try 
        block_size; 
    catch; 
        block_size=2 * nwt; end;
end;

try isempty(j2);
    if j2<=j1+2;  j2=j1+2; end
catch;
    try N>0;
        try
            j2;
        catch;
            j2=log2(N)-4; disp(['Set j2=',num2str(j2),' (default)']);  end;
        try
            block_size;
        catch;
            block_size=floor(N/32); end;
    catch;
        error('Cannot determine j2 or Blocklength: Need data length');
    end;
end


