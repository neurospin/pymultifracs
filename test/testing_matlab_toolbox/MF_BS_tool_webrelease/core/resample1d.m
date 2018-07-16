function [estimates, LEst]=resample1d(X, param_bs, param_est, TTsave, verbosity, FIGNUM, BSid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Calculate Bootstrap (BS) resamples of an estimate and relevant statistics
%%
% function [estimates, LEst]=resample1d(X, param_bs, param_est, TTsave, verbosity, FIGNUM, BSid);
%% 
% INPUT VARIABLES
%   X         - One dimensional Sample
%   param_bs   - Structure containing BS relevant parameters:
%          param_bs.n_resamp_1    : # primary BS resamples (for empirical pdf)     [default: 999]
%          param_bs.n_resamp_2    : # secondary BS resamples (var and adjusted)    [default: 50]
%          param_bs.block_size : block length for moving blocks BS (1 for ordinary BS) [default: 1]
%          param_bs.ci_method      : array with numbers from 1:8 determining which Confidence interval
%                                to calculate (e.g. [3] or [1 2 5] - multiple choices possible)
%                                                                               [default: 1:6]
%                                  1 - Normal approximation CI      [recommended n_resamp_2 = 50]
%                                  2 - Basic BS CI                  [recommended n_resamp_2 = 999]
%                                  3 - Percentile BS CI             [recommended n_resamp_2 = 999]
%                                  4 - Studentised BS CI            [recommended n_resamp_2 = 999]
%                                  5 - Adjusted Basic BS CI         [recommended n_resamp_2 = 250]
%                                  6 - Adjusted Percentile BS CI    [recommended n_resamp_2 = 250]
%   param_est  - Structure containing Estimation relevant parameters:
%          param_est.fhandle    : Estimator function handle (string, e.g. param_est.fhandle = 'mean')
%          param_est.param      : Object (structure, array, ...) of parameters needed by the Estimator 'fhandle' 
%                                If not needed (e.g. for 'mean': either omit or set param_est.param=struct('param',{}))
%   TTsave    -  0: save double bootstrap estimates t** only when needed (i.e., when ci_method is 5 or 6)
%             -  1: save double bootstrap estimates t** whenever they are calculated (i.e, when ci_method is 4, 5 or 6)
%                                                                   [default: 0]
%          verbosity             : verbosity level                                [default: 0]
%                                  0 - no display; batch mode
%                                  1 - display of estimate and BS STD estimate
%                                  2 - estimate, BS STD estimate and input parameters
%                                  3 - estimate, BS STD estimate and input parameters
%                                      QQ plot of input sample
%          FIGNUM              : Output Figure Number               [default: 1000]
%          BSid  (optional)    : BSid.B1(1:B1,1:Nsamp), BSid.B2(1:B1,1:B2,1:Nsamp) 
%%
% OUTPUT VARIABLES
%   estimates     - Structure containing estimates and BS distributions
%          estimates.t     :   Estimate
%          estimates.stdt  :   BS std estimate of std(t)
%          estimates.T     :   array of BS resample estimates t* of t                   [only for ci_method 4]
%          estimates.stdT  :   array of BS std estimate of std(t*)                      [only for ci_method 4]
%          estimates.TT    :   matrix of BS re-resample estimates t** of t*             [only for ci_method 5 and 6]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   USAGE EXAMPLE :
%   Batch mode calculation of BS Statistics for Percentile, Studentised CI or Test on variance of random sample X
%       param_bs  = struct('n_resamp_1', 999, 'n_resamp_2', 50, 'block_size', 1, 'ci_method', [3 4]);
%       param_est = struct('fhandle', 'var', 'param', []);
%       TTsave=0; verbosity=0;
%       [estimates]=BSsample(X, param_bs, param_est, TTsave, verbosity);
% 
% Herwig Wendt, Lyon, 2006 - 2008
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%
% NOTE: return of fhandle must be either scalar or a "row vector": cell2mat(out) --> [1xNest] 
%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK FOR CORRECT USE
%%%%%%%%%%%%%%%%%%%%%%%%%
% Check use of function
if nargout>2
    error('TOO MANY OUTPUT PARAMETERS.');
end
if nargin==3
    TTsave=0; verbosity=0; FIGNUM=1000;
elseif nargin==4
    verbosity=0; FIGNUM=1000;
elseif nargin==5
    FIGNUM=1000;
elseif (nargin<4)|(nargin>7)
    error('WRONG NUMBER OF INPUT PARAMETERS.');
end
if (nargin==7)&verbosity
    useid=1;
    disp('in ndimBSresample: using given bootstrap indices');
else
    useid=0;
end

%- RFL 2016-04-08: all this tests should be done in check_params now.

% Check input arguments
%param_bs=struct('n_resamp_1',B1,'n_resamp_2',B2, 'block_size', Block, 'ci_method', ci_method, 'verb', verbosity);
%-InputError1='param_bs must be a structure with elements:\n   param_bs.n_resamp_1 \n   param_bs.n_resamp_2\n   param_bs.block_size\n   param_bs.ci_method\n   param_bs.verb (optional)\n';


%-try  NB=isreal(TTsave);      if ~NB; TTsave=0; end;  catch TTsave=0; end
%-try  NB=isreal(verbosity);      if ~NB; verbosity=0; end;  catch verbosity=0; end
%-try  NB1=isreal(param_bs.n_resamp_1);      if ~NB1; param_bs.n_resamp_1=999; end;  catch error(InputError1, NB1); end
%-try  NB2=isreal(param_bs.n_resamp_2);      if ~NB2; param_bs.n_resamp_2=50;  end; catch error(InputError1, NB2); end
%-try  NB3=isreal(param_bs.block_size);   if ~NB3; param_bs.blocklength=1; end;  catch error(InputError1, NB3); end
%-try  NB4=isreal(param_bs.ci_method);        if ~NB4; param_bs.ci_method=[1:6];end;  catch error(InputError1, NB4); end
%-try CHR=ischar(param_est.fhandle); catch error('The structure param_est must contain a field  param_est.fhandle with a valid function handle'); end
%-if ~CHR; error('The function handle param_est.fhandle is not valid'); end
%-try NB3=isreal(X); catch error('Something is wrong - unable to read input data X.'); end
%-if ~NB3; error('The input data X must be numeric'); end
%-[l1,l2]=size(X);
%-if min(l1,l2)~=1; error('Only 1-dimensional data X allowed.'); end;

%%%%%%%%%%%%%%%%%%%%%%%%%%
% SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get estimation parameters
N=length(X);                % sample length
FHandle=param_est.fhandle;   % Estimator name

%FHandle=fhandle);       % Function handle for Estimator
%NOUT=nargout(FHandle);           % Number of Output Arguments of Estimator

% Check if Estimator needs extra parameters
% fparam=1;
% if length(fieldnames(param_est))==1
%     fparam=0;
% elseif isempty(param_est.param)
%     fparam=0;
% end

EstFun=param_est.estimate_select;
estimate_select=bin2dec(num2str(EstFun));

if (estimate_select == 1)||(estimate_select == 4); NOUT=1;
elseif (estimate_select == 2)||(estimate_select == 5); NOUT=2;
elseif (estimate_select == 3)||(estimate_select == 6); NOUT=3;
else; NOUT=4; end

% Check which Bootstrap statistics are to be calculated
ci_method=param_bs.ci_method; CIdisp=[]; Hstat=[];
if find(ci_method==1); NOR=1; else NOR=0; end
if find(ci_method==2); BAS=1; else BAS=0; end
if find(ci_method==3); PER=1; else PER=0; end
if find(ci_method==4); STU=1; else STU=0; end
if find(ci_method==5); BASADJ=1; else BASADJ=0; end
if find(ci_method==6); PERADJ=1; else PERADJ=0; end

% need double bootstrap values for Adjusted Methods
if TTsave~=0; TTsave=1; end
if ~(STU|BASADJ|PERADJ); TTsave=0; end
if PERADJ|BASADJ; TTsave=1; end
% Bootstrap parameters
w=0.5;          % window width (data length fraction) for Variance Function smoothing (Variance stabilising transformation CI)
B1=param_bs.n_resamp_1;        % number of primary bootstrap resamples
B2=param_bs.n_resamp_2;        % number of bootstrap resamples for variance estimates (used for Normal, Studentised, Variance Stabilising Transformation)
Block=param_bs.block_size;  % Block length for moving blocks resampling
if useid
    Block=1;
end

% initialize Block bootstrap
CIRCULAR=1;
if CIRCULAR
    N_range=N;
else
    N_range=max(N-Block+1,1);
end
N_BS=ceil(N/Block);
N_resample=N_BS*Block;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    ESTIMATE AND BOOTSTRAP ESTIMATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%
t=cell(1,NOUT);
% if ~fparam
%     %[t_out{:}]=feval(fhandle,X);
%     [t{:}]=FHandle(X);
% else
    %[t{:}]=FHandle(X,param_est.param);
    [t{:}]=FHandle(X,param_est);
%end
% get length of estimates
for ii=1:NOUT
    LEst(ii)=length(t{ii});
end
t=cell2mat(t);

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bootstrap distribution estimates G(t)=Pr(T<t|F1) of Pr(T<t|F0) 
% used for :    basic 
%               percentile 
%               studentised 
%%%%%%%%%%%%%%%%%%%%%%%%%%
if NOR|BAS|PER|STU|BASADJ|PERADJ
    T=cell(B1,NOUT);
    if STU|BASADJ|PERADJ
        TTT=cell(B2,NOUT);
    end

    if Block>1
        % initialize blocks of indices
        bx=0:N-1;
        try % fast but memory intensive
            bx=0:N-1;
            BX=repmat(bx,Block,1);
            addit=repmat([0:Block-1]', 1,N);
            BX=BX+addit;
            BX=mod(BX,N)+1;
        catch % slow but memory save
            BX=[];bx=1:N;
            for bl=1:Block
                BX=[BX; bx];
                bx=[bx(end) bx(1:end-1)];   % create overlapping blocks
            end
        end
    end

    for b=1:B1
        if Block==1
            if ~useid
                index = fix(rand(1,N)*N)+1 ;
            else
                index = BSid.B1(b,:);   % use BS indeces given
            end
            Xsample=X(index);
%             if ~fparam
%                 [T{b,:}]=FHandle(Xsample);
%             else
                %[T{b,:}]=FHandle(Xsample, param_est.param);
                [T{b,:}]=FHandle(Xsample, param_est);
%             end
        else  % block bootstrap
            index = fix(rand(1,N_BS)*N_range)+1 ;
            tempid=reshape(BX(:,index),1,[]);
            Xsample=X(tempid);
%             if ~fparam
%                 [T{b,:}]=FHandle(Xsample);
%             else
                %[T{b,:}]=FHandle(Xsample,param_est.param);
                [T{b,:}]=FHandle(Xsample,param_est);
                %            [T{b,:}]=FHandle(reshape(BX(:,index),1,N_resample),param_est.param);
%             end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        % Nested Bootstrap : Estimate standard deviation of bootstrap resample T
        % used for:     studentised
        %               variance stabilising transformation
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        if STU|BASADJ|PERADJ
            for bb=1:B2
                if Block==1
                    if ~useid
                        index = fix(rand(1,N)*N)+1 ;
                    else
                        index = squeeze(BSid.B2(b,bb,:))';   % use BS indeces given
                    end
                    Xresample=Xsample(index);
%                     if ~fparam
%                         [TTT{bb,:}]=FHandle(Xresample);
%                     else
%                         [TTT{bb,:}]=FHandle(Xresample, param_est.param);
                        [TTT{bb,:}]=FHandle(Xresample, param_est);
%                     end
                else  % block bootstrap
                    % ---> new: resample from blocks
                    index2 = fix(rand(1,N_BS)*N_BS)+1 ;
                    tempid=reshape(BX(:,index(index2)),1,[]);
                    Xresample=X(tempid);
                    % <--- new 12/12/06 HW Lyon
%                     if ~fparam
%                         [TTT{bb,:}]=FHandle(Xresample);
%                     else
                        %[TTT{bb,:}]=FHandle(Xresample, param_est.param);
                        [TTT{bb,:}]=FHandle(Xresample, param_est);
%                     end
                end
            end
            if STU
                stdT(b,:)=std(cell2mat(TTT)); % WORKS
            end
            if TTsave; TT(:,b,:)=cell2mat(TTT); end
        end
    end
    T=cell2mat(T);
    stdt=std(T);
else
    T=NaN(1, B1);
    stdt=NaN;
end
if ~(STU)
    stdT=NaN(1, B1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save bootstrap estimates and distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%
if TTsave
    estimates=struct('t', t, 'stdt', stdt, 'T', T, 'stdT', stdT, 'TT', TT);
else
    estimates=struct('t', t, 'stdt', stdt, 'T', T, 'stdT', stdT);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DISPLAY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if verbosity>=1
    % Parameters
    disp('*****************************************');
    if verbosity>=2    
    disp(' --- PARAMETERS --- ');
    disp(['samples:              ',num2str(N)]);
    disp(['primary resamples:    ',num2str(param_bs.n_resamp_1)]);
    disp(['secondary resamples:  ',num2str(param_bs.n_resamp_2)]);
    disp(['block length:         ',num2str(param_bs.block_size)]);
    disp(['Estimator:            ',param_est.fhandle]);
    disp('*****************************************');
    end
    % Estimates
    disp(' --- ESTIMATES --- ');
    disp(['Estimate t:           ',num2str(estimates.t)]);
    disp(['Bootstrap std(t):     ',num2str(estimates.stdt)]);
    disp('*****************************************');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if verbosity>=3
    figure(FIGNUM); clf; 
    subplot(211); hist(X, max(10, ceil(N/25))); xlabel('X'); ylabel('Frequency'); title('Input Sample X');
    subplot(212); qqplot(X); ylabel('Quantiles of X');
end