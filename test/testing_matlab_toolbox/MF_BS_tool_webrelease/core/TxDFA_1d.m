function [coef] = TxDFA_1d(data, poly_ord, p)
%
% Computes the (generalized) MFDFA multiresoultion quantities at dyadic
% scales.
%
% Input:
%    data: input data vector
%    poly_ord: order of polynomial to fit
%    p: p-exponent. p = 2 (default) is the traditional MFDFA exponent.
%
% RFL
% August 2013

% Input checking
if ~isvector (data)
    error ('Data should be a vector.')
end

if poly_ord < 0
    error ('Polynomial order should be greater or equal than 0')
end

if p < 0
    error ('p exponent should be greater or equal than 0')
end


if nargin < 3
    p = 2;
end

% Range of scales over which multiresolution quantity is computed.
% Below j1, polynomial can't be fitted reliably. 2^j2 is the largest dyadic
% scale possible.
j1 = ceil (log2 (poly_ord)) + 1;
j2 = floor (log2 (length (data)));


for j = 1 : j2
    % Scale index for output vector:
    jid = j;% - j1 + 1;

    ncoefs = length (data);

    % Important to have a row vector here, otherwise flexEstFun_MFA doesn't
    % work.
    value = zeros (1, ncoefs);
    xx = 1 : (2 ^ j);

    for k = 1 : floor (ncoefs ./ 2 ^j)
        idx = ((k - 1) * (2 ^ j) + 1) : (k * (2 ^ j));
        block = data(idx);

        % If j < j1, don't detrend but compute quantity anyway to preserve
        % dimension of output vector.
        % Note that these scales should never be used in the regressions.
        if j >= j1
            % mi_polyfit.m is the same that polyfit.m, but I've commented a
            % test on the condition number of a matrix that was responsible
            % for 85% of the time spent in the routine.
            poly = polyfit (xx, block, poly_ord);
            tmp = block - polyval (poly, xx);
        else
            tmp = block;
        end
        value(idx) = tmp;
    end

    % Output: I repeat info in all fields of output structures so as not to
    % break anything in client code.
    coef.value{j} = value;
    coef.nj(j) = ncoefs;
    coef.xpos{j} = 1 : ncoefs;
    % Bogus value:
    coef.zp = 0;
end
