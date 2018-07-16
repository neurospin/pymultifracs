function [varargout] = ...
               select_scaling_range( logstat, jlimit, idx, wtype, ...
                                     varj ) 
% function [varargout] = ...
%               select_scaling_range( logstat, jlimit, idx, wtype, ...
%                                     varj ) 
%
% Computes the change in goodnes-of-fit estimators of structure functions
% with the scaling range used for the regression.
%
% It finds the range of scales [jleft jright] where the regression is
% optimal maximizing the cost function Q(j1,j2), 
% where Q is a goodness-of-fit statistic.
%
% INPUTS:
%        - logstat: structure with fields:
%                - est logsf: matrix (nq x nj) with the structure functions
%                  for all q.
%                - BS.estB: (nbs x nq x nj) bootstrap estimates of structure 
%                  functions.
%                - BS.estBB: (nbs x nbs x nq x nj) double bootstrap estimates of
%                  structure functions.
%        - jlimit: matrix that indicates the search limits for j1 and j2, in
%        the format [ j1_min j1_max ; j2_min j2_max]
%        - idx: orders that are used for joint scaling range selection. This
%        is used to index the 'nq' dimension of the matrices
%        - wtype: weights of the regression.
%                 0: unweighted (wj=1)
%                 1: wj=1/nj
%                 2: BS estimated variance
%        - varj: (nq x nj) estimated variance.
% OUTPUTS:
%        - jleft: left scale of detected range
%        - jright: right scale of detected range
%        - gof: structure with fields for R(q,j1,j2) Q(q,j1,j2) and C(j1,j2).
%        - slope: (nq x 1) slope of linear regressions in range [jleft, jright]
%        -intercept: (nq x 1) intercept of linear regressions in range [jleft,
%        jright]
%        - Q: quality of regression for each q.
%
% version 15
%
% RFL, 18 February 2013
% Modif 07 April 2016
%    - Adapt interface to toolbox.
%    - Remove gamma business from interface.

%% INITIALIZATION

if nargin < 2; jlimit = [1 inf; 1 inf]; end
if nargin < 3 || isempty (idx); idx = 1 : length (logstat.param_est.q);end 
if nargin < 4; wtype = 0; end
if nargin < 5;  varj = []; end

if wtype == 2 || isempty (varj)
    varj = logstat.BS.Vest;
end

useGamma = 0;  % Disable gamma fit
K = 0;  % Useless old parameter. Should be removed.
alphaGamfit = 0.05; 

logsf = logstat.est(idx,:);
if isfield (logstat.BS, 'estB')
    logsfBS = logstat.BS.estB(:, idx, :);
else
    logsfBS = [];
end
if isfield (logstat.BS, 'estBB')
    logsfBBS = logstat.BS.estBB(:, :, idx, :);
else
    logsfBBS = [];
end

[nq, nj] = size (logsf);

computeQ = 1;
try 
    logsf.Q;
    computeQ = 0; 
    nq = size( logsf.Q, 1 );
    nj = size( logsf.Q, 2 );
catch 
end

  
j = (1 : nj);

[a,b] = size(jlimit);
% FIXME This conditional is no longer needed.
if a == 1 || b == 1
    % Force jmin to be at least 1:
    jmin = max( 1, jlimit(1) );
    % Force jmax to be at most nj:
    jmax = min( nj, jlimit(2) );
    
    j1max = nj - 1;
    j2min = jmin;
else
    jmin = max( 1, jlimit(1,1) );
    % Force jmax to be at most nj:
    jmax = min( nj, jlimit(2,2) );
    
    j1max = min(nj - 1, jlimit(1,2));
    j2min = jlimit(2,1);
end

[nbs,~,~] = size( logsfBS );
[nbbs,~,~,~] = size( logsfBBS );

if nbs == 0
    useBS = 0;
else
    useBS = 1;
end

if nbbs == 0
    useBBS = 0;
else
    useBBS = 1;
end


if wtype == 0  % unweighted regression
    wvar = ones( 1, nj );
elseif wtype == 1   % sigmaj = nj
    wvar = 1 ./ 2.^( nj:-1:1 );
elseif wtype == 2    %sigma = varj
    wvar = varj;
else
    wvar = ones( 1, nj );
end


%% COMPUTATION OF COST FUNCTION

if computeQ
gof.rss   = nan( nq, nj, nj );
gof.Q     = nan( nq, nj, nj );
gof.C     = nan( nj );
gof.lines = nan( nq, nj, nj, 2 );
gof.phat  = nan( nq, nj, nj, 2 );
gof.phatIC = nan( nq, nj, nj, 2, 2 );
gof.nlogL = nan( nq, nj, nj );
gof.chi2pval = nan( nq, nj, nj );
smallCounts = 0;


for j1 = jmin : j1max
    j2min = max(j1+2, j2min);
    for j2 = j2min : jmax
        C = 0;
        for q = 1 : nq
        
            if wtype == 2
                wvar = varj(q,:);
            end
                  
            % Selection of octaves, variances and estimates according to the
            % current scaling range:
            jj = j(j1:j2);
            wvarjj = wvar(j1:j2);
            
            % Precalculations for linear regression in this range of scales:
            S0 = sum( 1 ./ wvarjj );
            S1 = sum( jj ./ wvarjj );
            S2 = sum( jj.^2 ./ wvarjj );
            wjj = (S0*jj - S1) ./  (S0*S2-S1^2) ./ wvarjj;
            vjj = (S2 - jj*S1) ./  (S0*S2-S1^2) ./ wvarjj;
        
            %logsfjj = logsf(q,j1:j2)/abs(qq(q));
            logsfjj = logsf(q,j1:j2);
            
            % Linear regression:   logsf = a*j + b
            a = wjj * logsfjj';  % a = sum( wjj.*logsfjj );  
            b = vjj * logsfjj';  % b = sum( vjj.*logsfjj );  
            
            rss = sum( ((logsfjj - a*jj-b).^2)./wvarjj );
            gof.rss(q,j1,j2) = rss;            
            %========= Computation of GOF estimators ======================
            
            if ~useBS
            % -- a) Chi^2 GOF, following veitch, abry, taqqu 2002
                Q = 1 - chi2cdf( rss, j2-j1-1 );
        
            else   % if useBS
                % -- b) Distribution estimated from BS resamples
                njj = length(jj);
                logsfBSjj = squeeze( logsfBS( :, q, jj ) );  %  nbs x length(jj)
                A = logsfBSjj * wjj';   % nbs x 1
                B = logsfBSjj * vjj';
                JJ = repmat( jj, nbs, 1 );
                
                %-- Double bootstrap estimate of variance for each primary bootstrap resample: --
                if useBBS
                    WVARJJ = zeros( nbs, njj );
                    for ib = 1 : nbs
                        WVARJJ(ib,:) = var( squeeze( logsfBBS( :, ib, q, jj ) ) );
                    end
                else
                    WVARJJ = repmat( wvarjj, nbs, 1 );
                end
                %----
                
                AA = repmat( A, 1, njj );
                BB = repmat( B, 1, njj );
                RSS = sum( ( ( logsfBSjj - AA .* JJ - BB ).^2 ) ./ WVARJJ, 2);
                
                if useGamma
                    % -- b.1) Gamma distribution estimated from BS resamples
                    [phat, ic] = gamfit( RSS, alphaGamfit );
                    Q = 1 - gamcdf( rss, phat(1), phat(2) );
                    gof.phat( q, j1, j2, : ) = phat;
                    gof.phatIC( q, j1, j2, :, : ) = ic;
                    gof.nlogL( q, j1, j2 ) = gamlike( phat, RSS );
                    
                    % Chi2 goodness-of-fit:
                    [~,pval] = chi2gof( RSS, ...
                                        'cdf', @(x)gamcdf(x,phat(1),phat(2)), ...
                                        'nparams', 2 );
                    gof.chi2pval(q, j1, j2 ) = pval;
                    
                else
                    % -- b.2) Empirical distribution
                    RSS = sort( RSS );
                    Q = 1 - sum( RSS < rss  ) / nbs;
                end
            end
            
            gof.Q( q, j1, j2 ) = Q;
            gof.lines( q, j1, j2, : ) = [a b];

            %==============================================================
        end
    end
end


% $$$ if smallCounts > 0
% $$$     fprintf( ['Warning: %g bins with less than 5 counts were found in the ' ...
% $$$               'chi-squared goodness-of-fit test.\n'], smallCounts )
% $$$ end

else % ~computeC
    gof = logsf;
end

j2 = repmat( 1:nj, nj, 1 );
j1 = repmat( (1:nj)', 1, nj );
gof.C = squeeze(sum( gof.Q, 1 )) / nq + K * (j2 - j1) / (nj - 2);

%% MAXIMIZATION

% Maximum m might not be unique. In that case, return indices of smallest j1
% and then largest j2.
%-[m, Jmax] = max( gof.C ); %Jmax: row with the index of the first maximum for each column
%-[~, jright] = max( m(end : -1 : 1) );  % Get last index case of repetition
%-jright = length (m) - jright + 1;
%-jleft = Jmax( jright );

[jj1, jj2] = ind2sub (size (gof.C), find (gof.C == max (max (gof.C))));
[~, idx_max] = max (jj2 - jj1);
jleft = jj1(idx_max);
jright = jj2(idx_max);

Q = gof.Q( :, jleft, jright );

%% REGRESSION IN THE OPTIMAL RANGE

% Compute only if asked:
if nargout > 4
    slope = zeros(nq,1);
    intercept = slope;
    for q = 1 : nq  
        if wtype == 2
            wvar = varj(q,:);
        end
            
        jj = j(jleft:jright);
        wvarjj = wvar(jleft:jright);
        logsfjj = logsf(q,jleft:jright);
        
        S0 = sum( 1 ./ wvarjj );
        S1 = sum( jj ./ wvarjj );
        S2 = sum( jj.^2 ./ wvarjj );
        wjj = (S0*jj - S1) ./  (S0*S2-S1^2) ./ wvarjj ;
        vjj = (S2 - jj*S1) ./  (S0*S2-S1^2) ./ wvarjj ;
        
        
        % Linear regression:   logsf = a*j + b
        slope(q) = sum( wjj.*logsfjj );
        intercept(q) = sum( vjj.*logsfjj );
    end
end

%===============================================================================
% OUTPUT
%===============================================================================

if nargout >= 1
    varargout{1} = jleft;
end
if nargout >= 2
    varargout{2} = jright;
end
if nargout >= 3
    varargout{3} = Q;
end


end  % function select_scaling_range
