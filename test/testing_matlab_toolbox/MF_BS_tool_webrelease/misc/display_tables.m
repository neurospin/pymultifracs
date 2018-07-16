function display_tables (logstat, est, stat)
% Displays tables of estimates, confidence intervals and test results.
%
% February, 2016

    %===============================================================================
    %--- SETUP
    %===============================================================================

    %--- Determine which methods were used ----
    flag_DWT = isfield (logstat, 'DWT');
    flag_LWT = isfield (logstat, 'LWT');

    if ~flag_DWT | ~flag_LWT
        error ('Either DWT or LWT must be computed')
    end

    if isfield (logstat.DWT, 'param_est')
        param_est = logstat.DWT.param_est;
    else
        param_est = logstat.LWT.param_est;
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
        
        if isfield (stat.DWT, 'param_test')
            param_test = stat.DWT.param_test;
        elseif isfield (stat.LWT, 'param_test')
            param_test = stat.LWT.param_test;
        else
            param_test = [];
        end
    else
        doBS = false;
    end

    %--- Determine what estimates where computed ---
    [ZQ, DH, CP] = which_estimates (param_est.estimate_select);
    [idx_zq, idx_dq, idx_hq idx_cum] = get_estimate_id (param_est);


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
    num_method = length (meth_name);

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

    ci_name=cell(1,8);
    ci_name{1}='NOR ';
    ci_name{2}='BAS ';
    ci_name{3}='PER ';
    ci_name{4}='STU ';
    ci_name{5}='ADJB';
    ci_name{6}='ADJP';

    test_name=cell(1,4);
    test_name{1}=' A: t>tnull';
    test_name{2}=' A: t<tnull';
    test_name{3}='|t-tnull|=0';
    test_name{4}=' t-tnull=0 ';

    pad_with = 2 + length (num2str (max (param_est.q)));
    est_name_func{1} = @(x) sprintf (sprintf ('zeta(q = %% %ig)', pad_with), param_est.q(x));
    est_name_func{2} = @(x) sprintf (sprintf ('D(q = %% %ig)'   , pad_with), param_est.q(x));
    est_name_func{3} = @(x) sprintf (sprintf ('h(q = %% %ig)'   , pad_with), param_est.q(x));
    est_name_func{4} = @(x) sprintf ('cum %i', x);
    
    %===============================================================================
    %--- ESTIMATES
    %===============================================================================
    fprintf ('********** * * * * * * * * * * * * * * * * * * * * * * * * * * * * **********\n');
    for im = 1 : num_method
        print_header_raw ();

        fprintf ('  Estimate');
        if doBS
            fprintf (' ( std* )');
        end
        fprintf ('\n');
        if CP
            fprintf ('-----------------------------------\n');
            print_estimates_raw (idx_cum, est_name_func{4});
        end
        if ZQ
            fprintf ('-----------------------------------\n');
            print_estimates_raw (idx_zq, est_name_func{1});
        end
        if DH
            fprintf ('-----------------------------------\n');
            print_estimates_raw (idx_dq, est_name_func{2});
            print_estimates_raw (idx_hq, est_name_func{3});
        end
        fprintf ('\n');
    end  % for im = 1 : num_method

    %===============================================================================
    %--- CONFIDENCE INTERVALS
    %===============================================================================
    if doBS && param_bs.CI
        fprintf ('********** * * * * * * * * * * * * * * * * * * * * * * * * * * * * **********\n');
        for im = 1 : num_method
            print_ci_header_raw ();
            for ci_meth = param_bs.ci_method
                print_ci_method_header ();
                if CP
                    fprintf ('-----------------------------------\n');
                    print_ci_raw (idx_cum, est_name_func{4});
                end
                if ZQ
                    fprintf ('-----------------------------------\n');
                    print_ci_raw (idx_zq, est_name_func{1});
                end
                if DH
                    fprintf ('-----------------------------------\n');
                    print_ci_raw (idx_dq, est_name_func{2});
                    print_ci_raw (idx_hq, est_name_func{3});
                end
                fprintf ('\n');
            end  % for ci_meth = param_bs.ci_method
            fprintf ('\n');
        end  % for im = 1 : num_method
    end % if param_bs.CI

    %===============================================================================
    %--- TESTS
    %===============================================================================
    if doBS && param_bs.TEST
        fprintf ('********** * * * * * * * * * * * * * * * * * * * * * * * * * * * * **********\n');
        for im = 1 : num_method
            print_test_raw ();
        end
    end % if param_bs.TEST

    fprintf ('********** * * * * * * * * * * * * * * * * * * * * * * * * * * * * **********\n');

    %===============================================================================
    %--- SMALL P WARNING
    %===============================================================================
    if flag_LWT
        p_disp = param_est.p;
        %- >----- RFL 13-05-2013 >-----
        %- The condition est.LWT.zp < 0 can only be evaluated if zp was
        %- computed. I make the evaluation conditional on plead_corr using
        %- short-circuit &&.
        %- if verbosity&(p_disp>p_est|est.LWT.zp<0)
        if p_disp > est.LWT.p0 || (param_est.plead_corr && est.LWT.zp < 0)
            fprintf ('********** * * * * * * * * * * * * * * * * * * * * * * * * * * * * **********\n');
            fprintf ('**                               WARNING                                   **\n');
            fprintf ('      NOT in Lp for chosen p = %.1g\n', p_disp);
            fprintf ('                          p0 = %.1g\n', est.LWT.p0);
            fprintf ('                     zeta(p) = %.1g\n', est.LWT.zp);
            fprintf ('      Use larger value for gamint or smaller value for p\n');
            fprintf ('********** * * * * * * * * * * * * * * * * * * * * * * * * * * * * **********\n');
        end
    end  % if flag_LWT

    %===============================================================================
    %--- ANCILLARY FUNCTIONS
    %===============================================================================
    function print_header_raw ()
        fprintf ('++++++++++++++++++++++++++++++++++\n');
        fprintf ('+++++     ESTIMATES %s\n', meth_str{im});
        fprintf ('++++++++++++++++++++++++++++++++++\n');
        fprintf ('[ j1 = %1g -- j2 = %2g -- wtype = %1g ]\n', param_est.j1, param_est.j2, param_est.wtype);
        fprintf ('Fractional Integration: gamma=%1.2f \n', param_est.gamint);
        fprintf ('-----------------------------------\n');
        fprintf ('p0  before int : p0=%1.2f \n', est.(meth_name{im}).p0noint);
        fprintf ('p0             : p0=%1.2f \n', est.(meth_name{im}).p0);
        fprintf ('zeta(p)        : zp=%1.2f \n', est.(meth_name{im}).zp);
        fprintf ('-----------------------------------\n');
        fprintf ('h_min d_X before int : h_min=%1.2f \n', est.(meth_name{im}).h_min - param_est.gamint);
        fprintf ('h_min d_X            : h_min=%1.2f \n', est.(meth_name{im}).h_min);
        fprintf ('-----------------------------------\n');
    end % funtion print_header_raw
    %-------------------------------------------------------------------------------
    function print_estimates_raw (idx, name_fun)
        for ie = 1 : length (idx)
            fprintf ('    %s = % 2.3f', name_fun (ie), est.(meth_name{im}).t(idx(ie)));
            if doBS
                fprintf(' (% 7.3f)', est.(meth_name{im}).BS.stdt(idx(ie)));
            end
            fprintf ('\n');
        end
    end  % function print_estimates_raw
    %-------------------------------------------------------------------------------
    function print_ci_header_raw ()
        fprintf ('+++++++++++++++++++++++++++++++++++++++++++++\n')
        fprintf ('+++++      CONFIDENCE INTERVALS %s\n', meth_str{im});
        fprintf ('+++++++++++++++++++++++++++++++++++++++++++++\n');
    end
    %-------------------------------------------------------------------------------    
    function print_ci_method_header ()
        fprintf('---  METHOD=%s   - ALPHA=%1.2f     ---\n',ci_name{ci_meth}, param_bs.alpha);
        fprintf('  Estimate       [CIlo    CIhi]  (estimate (std*))\n');
    end
    %-------------------------------------------------------------------------------    
    function print_ci_raw (idx, name_fun)
        for ie = 1 : length (idx)
            fprintf ('    %s : [ % 7.3f   % 7.3f ]  ( % 7.3f (% 7.3f) )\n', ...
                     name_fun (ie), ...
                     stat.(meth_name{im}).confidence{idx(ie)}{ci_meth}.lo, ...
                     stat.(meth_name{im}).confidence{idx(ie)}{ci_meth}.hi, ...
                     est.(meth_name{im}).t(idx(ie)), ...
                     est.(meth_name{im}).BS.stdt(idx(ie)));
        end
    end  % function print_ci_raw
    %-------------------------------------------------------------------------------
    function print_test_raw ()
        fprintf ('+++++++++++++++++++++++++++++++++++++++++++++\n')
        fprintf ('+++      CUMULANTS:        TESTS %s\n', meth_str{im});
        fprintf ('+++++++++++++++++++++++++++++++++++++++++++++\n')
        for ic = 1 : param_est.cum
            fprintf ('----- Cumulant %i   -   ALPHA=%1.2f     ---\n', ic, param_bs.alpha);
            fprintf ('Hnull: c_%1g =%2.4f\n', ic, param_test.null_param(ic))
            fprintf ('c_%1g = %2.4f  ( %2.4f )\n', ...
                     ic, ...
                     est.(meth_name{im}).t(idx_cum(ic)), ...
                     est.(meth_name{im}).BS.stdt(idx_cum(ic)));
            for type = param_test.null_type
                fprintf ('- Test Statistic %s        \n',  test_name{type});
                fprintf ('Method     Reject   P-value     \n');
                for ci_meth = param_bs.ci_method
                    fprintf ('%s :     %1g         %2.3f \n', ...
                             ci_name{ci_meth}, ...
                             stat.(meth_name{im}).significance{ic}{ci_meth}{type}.reject, ...
                             stat.(meth_name{im}).significance{ic}{ci_meth}{type}.plevel );
                end
            end
        end
    end  % function print_test_raw
    %-------------------------------------------------------------------------------
end  % function display_tables
