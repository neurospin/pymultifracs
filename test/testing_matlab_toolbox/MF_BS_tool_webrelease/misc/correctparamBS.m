function [param_bs] = correctparamBS(varargin)
% function [param_bs] = correctparamBS(varargin)
% write correct / non redundant bootstrap parameters for BSresample and
% ndimBSresample
% 2 possible usages:
%% A) [param_bs] = correctparamBS(param_bs)
%       checks bootstrap parameters and corrects them, if necessary
%% B) [param_bs] = correctparamBS(B1, B2, Blocklength, ci_method)
%       writes correct bootstrap parameters
%
% Herwig Wendt, Lyon, 2006 - 2008

%% STEP 1: READ PARAMETERS
if nargin==1
    STRUCT=1;
    temp_param_bs=varargin{1}; 
    InputError1=('Error: single input argument must be structure with 9 fields: B1, B2, Blocklength, ci_method');
    InputError2=('Error: param_bs must contain 4 fields: B1, B2, Blocklength, ci_method');
    try  fnames=fieldnames(temp_param_bs); catch error(InputError1, 0); end
    try  B1=getfield(temp_param_bs, fnames{1}); catch error(InputError2, 0); end
    try  B2=getfield(temp_param_bs, fnames{2}); catch error(InputError2, 0); end
    try  BlockLength=getfield(temp_param_bs, fnames{3}); catch error(InputError2, 0); end
    try  METHOD=getfield(temp_param_bs, fnames{4}); catch error(InputError2, 0); end    
elseif nargin==4
    STRUCT=0;
    B1=varargin{1};
    B2=varargin{2};
    BlockLength=varargin{3};
    METHOD=varargin{4};
else
    error('Wrong number of input arguments');
end

ci_method=[];
if find(METHOD==1); ci_method=[ci_method 1]; end
if find(METHOD==2); ci_method=[ci_method 2]; end
if find(METHOD==3); ci_method=[ci_method 3]; end
if find(METHOD==4); ci_method=[ci_method 4]; end
if find(METHOD==5); ci_method=[ci_method 5]; end
if find(METHOD==6); ci_method=[ci_method 6]; end

%% STEP 2: CHECK PARAMETERS
% block length
if BlockLength<1; BlockLength=1; end

param_bs = struct('n_resamp_1',B1,'n_resamp_2',B2, 'block_size', BlockLength, 'ci_method', ci_method);

% check if double bootstrap methods can be calculated
if ~(B2>1) 
    ci_method=ci_method(find(ci_method~=4));
    ci_method=ci_method(find(ci_method~=5));
    ci_method=ci_method(find(ci_method~=6));
    param_bs = struct('n_resamp_1',B1,'n_resamp_2',B2, 'block_size', BlockLength, 'ci_method', ci_method);
end

% Check if double bootstrap necessary
if ~(length(find(ci_method==4))|length(find(ci_method==5))|length(find(ci_method==6))); 
    B2=1; 
    param_bs = struct('n_resamp_1',B1,'n_resamp_2',B2, 'block_size', BlockLength, 'ci_method', ci_method);
end