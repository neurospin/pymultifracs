function q = build_q_log (q_min, q_max, n)
% function q = build_log_q (q_min, q_max, n)
% Builds q a convenient vector of q values for multifractal analysis, consisting of:
%    - \pm log-spaced values between q_min and q_max
%    - 0, \pm 1 and \pm 2,
%

if q_min <= 0 || q_max <= 0
    error ('q_min and q_max must be larger than 0')
end

q = logspace (log10 (q_min), log10 (q_max), n);
q = [q 1 2];
q = unique (sort ([q -q 0]));
