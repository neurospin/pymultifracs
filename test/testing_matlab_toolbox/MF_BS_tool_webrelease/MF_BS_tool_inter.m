%===============================================================================
% Class-based interface to MF_BS_tool
%
% RFL,
% February 2015
%===============================================================================

% Must derive from handle because otherwise self is passed by value and
% objects can't modify its parameters
% Instead of handle inherit from matlab.mixin.Copyable to inherit a copy
% function to make deep copies of objects.

classdef MF_BS_tool_inter < matlab.mixin.Copyable
    %===========================================================================
    properties (Access = public)
        % Estimation parameters
        method_mrq      = [1 2];   % Multiresolution quantity.
        nwt             = 3;       % Number of vanishing moments
        sym             = 0;       % Symmetrize wavelet 1d only)
        gamint          = 0;       % Pseudo-fractional integration order
        j1              = 1;       % Lower cutoff of scaling range
        j2              = 20;      % Upper cutoff of scaling range
        wtype           = 0;       % Regression type
        estimate_select = 111;     % Estimates to compute
        cum             = 2;       % Highest order of log cumulant
        q               = [-5 : 5];  % Moments to estimate
        p               = inf;     % Norm of p-leaders
        verbosity       = 1;       % Verbosity level
        fig_num         = 1;       % Figure number
        plead_corr      = 2;       % Correction term for p-leaders
        fhandle         = @flexEstFun_MFA;  % Handle of function to compute estimates
        num_threads     = 0;       % Number of threads to use (for mex only)
        j1_ini          = 1;       % Initial scale for computation of leaders
        type_exponent   = 0;       % Type of pointwise exponent
        delta_gam       = 0.05;    % Delta of fractional integration for oscillation exponent
        delta_p         = 0.05;    % Delta of p for lacunarity exponent

        % Bootstrap parameters
        CI            = 0;    % Compute confidence intervals
        TEST          = 0;    % Compute tests
        bs_type       = 0;    % Type of bootstrap.
        n_resamp_1    = 0;    % Number of primary bootstrap resamples
        n_resamp_2    = 0;    % Number of secondary bootstrap resamples
        block_size    = 10;    % Block size for block-bootstrap
        ci_method     = 3;    % Method for condifdence intervals
        alpha         = 0.05; % Significance level
        flag_bs_range = 0;    % Limit bootstrap to scaling range

        % Test parameters
        null_type  = 4;   % Null Hypothesis type
        null_param;       % Values of null hypothesis

        %Results of analysis
        est     = [];   % Final estimates
        logstat = [];   % Structure functions, log-cumulants
        stat    = [];   % Confidence intervals and tests
        mrq     = [];   % Multiresolution quantities.
    end  % public properties
    %===========================================================================
    methods  % Set and get methods
        function set.cum (self, cum)
        % Adapts default value of null_param, which depends on cum.
        % If null param was set to something different, does nothing
            self.cum = cum;
            if all (self.null_param == 0)
                self.null_param = zeros (1, cum);
            end
        end  % set.cum
    end  % set and get methods
    %===========================================================================
    methods (Access = public)
        %-----------------------------------------------------------------------
        function obj = MF_BS_tool_inter ()
        % Class contructor

            % add subfolders to path
            pathMF_BS_tool_singul;

           % Initialize properties that depend on others:
            obj.null_param = zeros (1, obj.cum);
        end
        %-----------------------------------------------------------------------
        function [varargout] = analyze (self, data)
        % Perform analysis with provided data or stored logscale quantities.

            self.pack_structs ();
            self.check_params ();

            if nargin < 2
            % Reuse internal logstat
            % TODO: should check compatibility of parameters (e.g that p or
            % momnul haven't changed)
                [self.est, self.logstat, self.stat] = ...
                    MF_BS_tool (self.logstat, self.param_est, self.method_mrq, ...
                                self.param_bs, self.paramTEST, ...
                                self.verbosity, self.fig_num);
            else
                % Call external MF_BS_tool with internal parameters.
                [self.est, self.logstat, self.stat, self.mrq] = ...
                    MF_BS_tool (data, self.param_est, self.method_mrq, ...
                                self.param_bs, self.paramTEST, ...
                                self.verbosity, self.fig_num);
            end

            if nargout >= 1
                varargout{1} = self.est;
            end
            if nargout >= 2
                varargout{2} = self.logstat;
            end
            if nargout >= 3
                varargout{3} = self.stat;
            end
            if nargout == 4
                varargout{4} = self.mrq;
            end
        end  % MF_BS_tool
        %-----------------------------------------------------------------------
        function [id] = get_zid (self)
        % Return indices of scaling exponents zeta(q).

            if self.check_if_computed ('zeta')
                id = 1 : length (self.q);
            else
                id = [];
            end
        end
        %-----------------------------------------------------------------------
        function [id] = get_Did (self)
        % Return indices of spectrum dimensions D(q).

            if self.check_if_computed ('D')
                ctrl = self.check_if_computed ('zeta');
                id = (1 : length (self.q)) + ctrl * length (self.q);
            else
                id = [];
            end
        end
        %-----------------------------------------------------------------------
        function [id] = get_hid (self)
        % Return indices of spectrum exponents h(q)

            if self.check_if_computed ('D')
                ctrl = self.check_if_computed ('zeta');
                id = (1 : length (self.q)) + (ctrl + 1) * length (self.q);
            else
                id = [];
            end
        end
        %-----------------------------------------------------------------------
        function [id] = get_cid (self)
        % Return indices of log-cumulants cp

            if self.check_if_computed ('c')
                ctrl = self.check_if_computed ('zeta') + ...
                       2 * self.check_if_computed ('D');
                id = (1 : self.cum) + ctrl * length (self.q);
            else
                id = [];
            end
        end
        %-----------------------------------------------------------------------
    end % public methods
    %===========================================================================
    properties (Access = private)
        param_est;
        param_bs;
        paramTEST;
    end  % private properties
    %===========================================================================
    methods (Access = private)
        %-----------------------------------------------------------------------
        function check_params (self)
        % Check the values of parameters, issue errors if invalid.
        %
            [self.param_est, self.param_bs, self.paramTEST] = ...
                check_parameters (self.param_est, self.param_bs, self.paramTEST);

            % Also change the class' public attributes so that they'r
            % consistent with the structs.
            self.unpack_structs ();
        end
        %-----------------------------------------------------------------------
        function pack_structs (self)
        % Store public properties in private structures

            % Estimation parameters
            self.param_est.nwt             = self.nwt;
            self.param_est.sym             = self.sym;
            self.param_est.j1_ini          = self.j1_ini;
            self.param_est.gamint          = self.gamint;
            self.param_est.delta_gam       = self.delta_gam;
            self.param_est.delta_p         = self.delta_p;
            self.param_est.type_exponent   = self.type_exponent;
            self.param_est.j1              = self.j1;
            self.param_est.j2              = self.j2;
            self.param_est.wtype           = self.wtype;
            self.param_est.estimate_select = self.estimate_select;
            self.param_est.cum             = self.cum;
            self.param_est.q               = self.q;
            self.param_est.p               = self.p;
            self.param_est.plead_corr      = self.plead_corr;
            self.param_est.fhandle         = self.fhandle;
            self.param_est.num_threads     = self.num_threads;

            self.param_est.method_mrq = self.method_mrq;
            self.param_est.verbosity  = self.verbosity;
            self.param_est.fig_num    = self.fig_num;

            % Bootstrap parameters
            self.param_bs.n_resamp_1    = self.n_resamp_1;
            self.param_bs.n_resamp_2    = self.n_resamp_2;
            self.param_bs.block_size    = self.block_size;
            self.param_bs.ci_method     = self.ci_method;
            self.param_bs.alpha         = self.alpha;
            self.param_bs.flag_bs_range = self.flag_bs_range;
            self.param_bs.bs_type       = self.bs_type;
            self.param_bs.CI            = self.CI;
            self.param_bs.TEST          = self.TEST;

            % Test parameters
            self.paramTEST.null_type  = self.null_type;
            self.paramTEST.null_param = self.null_param;
        end  % pack_structs ()
        %-----------------------------------------------------------------------
        function  unpack_structs (self)
        % Set public properties from private structures

            % Estimation parameters
            self.nwt             = self.param_est.nwt;
            self.param_est.sym   = self.sym;
            self.j1_ini          = self.param_est.j1_ini;
            self.gamint          = self.param_est.gamint;
            self.delta_gam       = self.param_est.delta_gam;
            self.delta_p         = self.param_est.delta_p;
            self.type_exponent   = self.param_est.type_exponent;
            self.j1              = self.param_est.j1;
            self.j2              = self.param_est.j2;
            self.wtype           = self.param_est.wtype;
            self.estimate_select = self.param_est.estimate_select;
            self.cum             = self.param_est.cum;
            self.q               = self.param_est.q;
            self.p               = self.param_est.p;
            self.plead_corr      = self.param_est.plead_corr;
            self.fhandle         = self.param_est.fhandle;
            self.num_threads     = self.param_est.num_threads;

            self.method_mrq = self.param_est.method_mrq;
            self.verbosity  = self.param_est.verbosity;
            self.fig_num    = self.param_est.fig_num;

            % Bootstrap parameters
            self.n_resamp_1    = self.param_bs.n_resamp_1;
            self.n_resamp_2    = self.param_bs.n_resamp_2;
            self.block_size    = self.param_bs.block_size;
            self.ci_method     = self.param_bs.ci_method;
            self.alpha         = self.param_bs.alpha;
            self.flag_bs_range = self.param_bs.flag_bs_range;
            self.bs_type       = self.param_bs.bs_type;
            self.CI            = self.param_bs.CI;
            self.TEST          = self.param_bs.TEST;

            % Test parameters
            self.null_type  = self.paramTEST.null_type;
            self.null_param = self.paramTEST.null_param;
        end  % pack_structs ()
        %-----------------------------------------------------------------------
        function [tf] = check_if_computed (self, x)
        % Check if the estimate x was computed or not.
        % Possible values for x are: 'zeta', 'D', 'c'
        % PRECONDITION: estimate_select has only three digits, with values 0 or 1

        % Return the corresponding digits of estimate_select in each case
            switch x
              case 'zeta'  % Third digit
                tf = mod (floor (self.estimate_select / 100), 10);
              case 'D'     % Second digit
                tf = mod (floor (self.estimate_select / 10), 10);
              case 'c'     % First digit
                tf = mod (self.estimate_select, 10);
              otherwise
                error ('Unknown estimate')
            end
        end
        %-----------------------------------------------------------------------
    end  % private methods
end  % classdef
