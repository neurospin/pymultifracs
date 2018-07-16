function [param_est, param_bs, paramTEST] = ...
        check_parameters (param_est, param_bs, paramTEST)
% function check_parameters (varargin)
%
% Checks if all necessary parameters for MF_BS_tool are defined and have
% proper values.
%
% Roberto Leonarduzzi, 2016

    % ----- Warning strings  ----------
    warn_str_no_max_num_thread_fun = ...
        ['I couldn''t use maxNumCompThreads to determine the number of ' ...
         'threads. This is likely because it has finally been deprecated' ...
         'I''ll default to 1 threads, but perhaps you should set a better' ...
         'value manually.'];

    % ----- Estimation parameters ----------

    % Auxiliary variables to keep the code tidy.
    aux_est_sel_p = @(x) in_set_p (x, [000 001 010 011 100 101 110 111]);
    aux_verb_p    = @(x) in_set_p (x, [0, 1, 10, 11]);
    qvec = build_q_log(0.01, 5, 10);

    % Try to use the maxNumCompThreads, disabling the warning telling it will
    % be deprecated. Display a warning when it doesn't work.
    try
        warning ('off', 'MATLAB:maxNumCompThreads:Deprecated')
        num_threads = maxNumCompThreads ();
        warning ('on', 'MATLAB:maxNumCompThreads:Deprecated')
    catch
        num_threads = 1;
        warning (warn_str_no_max_num_thread_fun)
    end

    % Make pre test of num_threads: if it is 0, assume it was set up by the
    % toolbox for automatic determinantion.
    % If it has any other unaccepted value, it will be caught by the text and the
    % corresponding warning will be shown.
    if param_est.num_threads == 0
        param_est.num_threads = num_threads;
    end

    % Content:
    %   variable name,  cell array of predicates, default value
    test_suite_est = {
        'method_mrq'      , {@vec_p, @(x) in_set_p(x, [1 2]) } , [1 2] ;
        'nwt'             , {@scal_p, @is_int_p, @positive_p  } , 3 ;
        'sym'             , {@scal_p, @(x) in_set_p(x, [0 1]) } , 0  ;
        'j1_ini'          , {@scal_p, @positive_p             } , 1 ;
        'gamint'          , {@scal_p                          } , 0 ;
        'delta_gam'       , {@scal_p, @positive_p             } , 0.1 ;
        'delta_p'         , {@scal_p, @positive_p             } , 0.1 ;
        'type_exponent'   , {@scal_p, @(x) in_set_p(x, 0 : 3) } , 0 ;
        'j1'              , {@vec_p, @is_int_p, @positive_p  } , 1 ;
        'wtype'           , {@scal_p, @(x) in_set_p(x, 0 : 2) } , 0 ;
        'estimate_select' , {@scal_p, aux_est_sel_p           } , 111 ;
        'cum'             , {@vec_p, @is_int_p, @positive_p   } , 1 : 3 ;
        'q'               , {@vec_p, @real_p                  } , qvec ;
        'p'               , {@scal_p             } , inf ;
        'verbosity'       , {@scal_p, aux_verb_p              } , 1 ;
        'fig_num'         , {@scal_p, @is_int_p, @positive_p  } , 1000 ;
        'plead_corr'      , {@scal_p, @(x) in_set_p(x, 0 : 2) } , 2 ;
        'num_threads'     , {@scal_p, @is_int_p, @positive_p  } , num_threads
        };

    param_est = apply_test_suite (param_est, test_suite_est, inputname (1));

    % Test for j2 now that existance of j1 is guaranteed
    test_j2 = {'j2', {@vec_p, @is_int_p, @positive_p, ...
                      @(x) geq_p(x, param_est.j1 + 2)}, inf};
    param_est = apply_test_suite (param_est, test_j2, inputname (1));



    % ----- Boostrap parameters ----------
    test_suite_bs = {
        'n_resamp_1'    , {@scal_p, @is_int_p, @nonnegative_p     } , 0 ;
        'n_resamp_2'    , {@scal_p, @is_int_p, @nonnegative_p     } , 0;
        'block_size'    , {@scal_p, @is_int_p, @positive_p        } , 1;
        'ci_method'     , {@vec_p, @(x) in_set_p(x, 1 : 6)        } , 3;
        'alpha'         , {@scal_p, @positive_p, @(x) leq_p(x, 1) } , 0.05;
        'flag_bs_range' , {@scal_p, @(x) in_set_p(x, 0 : 1)       } , 0;
        'bs_type'       , {@scal_p, @(x) in_set_p(x, 0 : 1)       } , 0;
        'CI'            , {@scal_p, @(x) in_set_p(x, 0 : 1)       } , 0;
        'TEST'          , {@scal_p, @(x) in_set_p(x, 0 : 1)       } , 0
                    };

    param_bs = apply_test_suite (param_bs, test_suite_bs, inputname (2));

    % ----- Test parameters ----------
    ncum = param_est.cum;
    null_param = zeros(1, ncum);

    test_suite_test = {
        'null_type'  , {@scal_p, @(x) in_set_p(x, 1 : 4) } , 4 ;
        'null_param' , {@vec_p, @(x) len_is_p(x, ncum)   } , null_param
                      };

    paramTEST = apply_test_suite (paramTEST, test_suite_test, inputname (3));



    % ----- SPECIAL TESTS ----------

    % Test for fhandle. Use mex function if available, else fall back to m file
    if exist ('compute_struct_func_mex', 'file') == 3  % file mex-file exists
        param_est.fhandle = @flexEstFun_MFA_mex;
    else
        param_est.fhandle = @flexEstFun_MFA;
        %warning (['I couldn''t find a compiled mex-file. '...
%          'I''ll use a (slower) m-file instead.\n'])
    end

end  % function check_parameters
%-------------------------------------------------------------------------------
function [params] = apply_test_suite (params, suite, name_orig)
    for i = 1 : size (suite, 1)
        params = aux_check_param (params, suite{i, :}, name_orig);
        % Note that cell array indexing produces comma separated list
    end
end
%-------------------------------------------------------------------------------
function [params] = aux_check_param (params, varname, require_list, default, name_orig)
% Checks if variable 'varname' is a field of 'params', and if it complies
% with the requirements in 'require_list'. If it doesnt, it is assigned the
% value 'default'.
%
% require_list is a list of handles to predicate functions. Each predicate
% must also return a second value with a descriptive message.

    if isfield (params, varname)
        for ireq = 1 : length (require_list)
            [complies, msg] = require_list{ireq} (params.(varname));
            if ~complies
                params.(varname) = default;
                warning (['Variable %s.%s doesn''t meet requirement: %s. '...
                          'I''ll use the default value.\n'], ...
                         name_orig, varname, msg);
                %Once it was replaced by the default it passes all tests, so I
                %exit the loop.
                break
            end
        end
    else
        warning (['Variable %s.%s doesn''t exist. ' ...
                  'I''ll use the default value.\n'], ...
                 inputname(1), varname);
        params.(varname) = default;
    end
end  % function aux_check_param
%-------------------------------------------------------------------------------
% Predicates
%---------------
function [tf, msg] = in_set_p (x, set)
% Precondition: x is scalar or vector
    if isscalar (x)
        tf = any (set == x);
    else
        tf = true;
        for i = 1 : length (x)
            tf = tf && any (set == x(i));
        end
    end

    format = repmat ('%i, ', 1, length (set));
    format = format(1 : end - 2);
    msg = sprintf (['x in [' format ']'], set);
end

function [tf, msg] = positive_p (x)
    tf = x > 0;
    msg = 'x > 0';
end

function [tf, msg] = nonnegative_p (x)
    tf = x >= 0;
    msg = 'x >= 0';
end

function [tf, msg] = is_int_p (x)
    tf = x == round (x);
    msg = 'x is integer';
end

function [tf, msg] = leq_p (x, lim)
    tf = x <= lim;
    msg = sprintf ('x <= %g', lim);
end

function [tf, msg] = geq_p (x, lim)
    tf = x >= lim;
    msg = sprintf ('x >= %g', lim);
end

function [tf, msg] = len_is_p (x, len)
    tf = length (x) == len;
    msg = sprintf ('Length of x is %i', len);
end

function [tf, msg] = vec_p (x)
    tf = isvector (x) && ~iscell (x);
    msg = 'x is vector';
end

function [tf, msg] = scal_p (x)
    tf = isscalar (x);
    msg = 'x is scalar';
end

function [tf, msg] = real_p (x)
    tf = isreal (x);
    msg = 'x must be real';
end
%-------------------------------------------------------------------------------
