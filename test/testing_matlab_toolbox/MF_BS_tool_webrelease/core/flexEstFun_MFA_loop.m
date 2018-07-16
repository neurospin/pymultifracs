function [Elogmuqj, f_Dq, f_hq, Cp] = flexEstFun_MFA_loop(Absdqk, param)
% function [Elogmuqj, f_Dq, f_hq, Cp]] = flexEstFun_MFA(Absdqk, param)
%
% Calculates Structure Functions (SF) and quantities for D(q), h(q) for a vector q (length(q)>1)
% Log Cumulants 1-5
%
% -- INPUT
%   Absdqk  -   |d_X(j,.)| or L_X(j,.), 
%   param   -   structure with parameters
%       param.q      : vector with moments q to calculate
%       param.cum    : highest order of Cumulant to calculate 
%       param.estimate_select : what is to calculate
%                       number xyz 
%                       x : SF zeta(q) [0 or 1]
%                       y : SF D(h) [0 or 1]
%                       y : SF Cp [0 or 1]
%                       e.g. param.estimate_select = 001 calculates only Cp, 
%                            param.estimate_select = 110 calculates SF zeta(q) and D(h)
%
%  Output that are not requested by param.estimate_select are returned empty
%
% Herwig Wendt, Lyon, 2006 - 2008

Absdqk = Absdqk(:)';

SORTOUT = 1;  % sort out small values
thresh = eps; % 1e-10;
if SORTOUT
    Absdqk = Absdqk(Absdqk >= thresh);
end

[ZQ, DH, CP] = which_estimates (param.estimate_select);

Elogmuqj = [];
f_Dq     = [];
f_hq     = [];
Cp       = [];

if ZQ || DH % if not only Cumulants
    q = param.q;
    for  kq=1:length(q)    %  Loop on the values of q
        if ZQ
            Elogmuqj(kq) = log2(mean(Absdqk.^q(kq)));    % Calculate  log[ S_q(j) ]    ( in L1 language )
        end
        
        if DH
            detail_kq = Absdqk.^q(kq);
            sum_detail_kq = sum(detail_kq);
            f_Dq(kq) = sum(detail_kq .* log2(detail_kq / sum_detail_kq )) / sum_detail_kq + log2(length(Absdqk));
            f_hq(kq) = sum(detail_kq .* log2(Absdqk)) / sum_detail_kq;
        end
    end
end

% Elogmuqj = [];
% f_Dq     = [];
% f_hq     = [];
% Cp       = [];
% 
% if ZQ || DH % if not only Cumulants
%     q = param.q;
%     % structure functions and Moments. f_Dq and f_hq WITHOUT LOOP
%     % This version is faster than the loop for length(q)>3 and length(Absdqk)>10 (~0.7 times the loop)
%     lenA = length (Absdqk);
%     lenq = length (q);
%     S = repmat (Absdqk, lenq, 1);
%     Q = repmat (q', 1, lenA);
%     detail_kq = S .^ Q;
% 
%     sum_detail_kq = sum(detail_kq, 2);
%     
%     if ZQ 
%         Elogmuqj = log2 (sum_detail_kq / length (detail_kq))';
%     end
%     
%     if DH
%         sum_detail_kq = sum(detail_kq, 2);
%         f_Dq = (sum (detail_kq .* log2 (detail_kq ./ ...
%                                         repmat (sum_detail_kq, 1, lenA)), ...
%                      2)./ sum_detail_kq + log2 (lenA))';
%         f_hq = (sum (detail_kq .* log2(S), 2) ./ sum_detail_kq)';
%     end
% end

if CP  % Cumulants
    Absdqk0 = log (Absdqk);
    Cp(1) = mean(Absdqk0) ;
    if param.cum > 1
        Absdqk = Absdqk0 .^ 2;
    end
    Cp(2) = mean (Absdqk) - Cp(1) ^ 2 ;
    if param.cum > 2
        Absdqk = Absdqk .* Absdqk0;

        Cp(3) = mean (Absdqk) - 3*Cp(2)*Cp(1) - Cp(1)^3 ;
        
    end
    if param.cum > 3
        Absdqk = Absdqk .* Absdqk0;
        Cp(4) = mean (Absdqk) - 4*Cp(3)*Cp(1) - 3*Cp(2)^2 - 6*Cp(2)*Cp(1)^2 - Cp(1)^4;
    end
    if param.cum > 4
        Absdqk = Absdqk .* Absdqk0;
        Cp(5) = mean (Absdqk) - 5*Cp(4)*Cp(1) - 10*Cp(3)*Cp(2) - 10*Cp(3)*Cp(1)^2 - 15*Cp(2)^2*Cp(1) - 10*Cp(2)*Cp(1)^3 - Cp(1)^5 ;
    end
end

