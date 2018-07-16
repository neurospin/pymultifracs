function p0 = p0est(coef,param_est)

nj = coef.nj;

NR=param_est.P0est.NR;
wtype=param_est.wtype;
gamint = param_est.gamint;
j1=param_est.j1;
j2=param_est.j2;

j2=min(length(coef.value),j2);

for j=1:length(coef.value)
    supcoef(j) = max (abs (coef.value{j}(:)));
end
h_min = MFA_BS_regrmat (log2 (supcoef), [], nj, wtype, j1, j2);
h_min_noint = h_min - gamint;

q0=param_est.P0est.q0; 
q1=param_est.P0est.q1;

 
% Actually compute the estimates, with and without integration
p0.noint = get_p0_gamma (0);
if gamint ~= 0
    p0.int = get_p0_gamma (gamint);
else
    p0.int = p0.noint;
end

%===============================================================================
% Helper nested functions
%-------------------------------------------------------------------------------
function p0qg = get_p0_gamma (gamma)
% Generic function to be reused for computation with and without integration 
    if h_min - gamint + gamma > 0   % Select h_min_noint for gamma=0, h_min ow.
        p0qg=inf;
    else
        % check smallest positive q
        ZQ0 = estimate_zq (q0, gamma);
        if ZQ0<=0  % not in Lp
            p0qg=-1; 
        else  % in some Lp
            ZQ1 = estimate_zq (q1, gamma);
            if ZQ1>=0  % in Lp with p>=q1
                p0qg=q1; 
            else   % in Lp with q0 <= p <= q1
                for k=1:NR
                    q01=mean([q0 q1]);
                    ZQ01 = estimate_zq (q01, gamma);
                    if ZQ01>0
                        q0=q01;
                    else 
                        q1=q01;
                    end
                end
                p0qg=q0;
            end
        end
    end
end  % function get_p0_gamma
%-------------------------------------------------------------------------------
function zq = estimate_zq (q, gamma)
    for j=j1:j2 
        SJQ(j)=log2(mean(abs(coef.value{j}(:).^q))); 
    end
    SJQ = SJQ + q * gamma * (1 : length (SJQ));
    zq = MFA_BS_regrmat(SJQ,[],nj, wtype, j1,j2);    

end  % function estimate_zq
%===============================================================================
    
end % function p0est