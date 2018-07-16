function [leader_osc] = singulPx(data, leader, param_est);

pp_min = 0.1;

gamint = param_est.gamint;
delta_gam   = param_est.delta_gam;
delta_p = param_est.delta_p;
type_exponent   = param_est.type_exponent;
p     = param_est.p;


switch type_exponent
    case 1 % Oscillation
        g2     = gamint + delta_gam;
        pp2    = p;
        ddelta = delta_gam;
        rm1    = 1; % remove +1 due to fact int
    case 2 % Lacunary
        g2 = gamint;
        pp2 = 1 / (1 / p + delta_p);
        ddelta = delta_p;
        rm1 = 0;
    case 3 % Cancellation
        g1     = gamint;
        g2     = gamint + delta_gam; 
        if isfinite (p)
            pp1    = 1 / (1 / p - g1);
            pp2    = 1 / (1 / p - g2);
        else
            pp2 = inf;
            pp1 = 1 / g2;
        end
        if pp1 < 0
            pp1 = pp_min;
        end
        if pp2 < 0
            pp2 = pp_min;
        end
        ddelta = delta_gam;
        rm1    = 1; % remove +1 due to fact int
end

param_est2        = param_est;
param_est2.gamint = g2;
param_est2.p     = pp2;

if type_exponent == 3
    param_est.gamint = g1;
    param_est.p     = pp1;
end

leader_osc = leader;

if ~leader.imagedata
    [~, leader2] = DxPx1d(data, param_est2);
    if type_exponent == 3
        [~, leader] = DxPx1d(data, param_est);
    end
else
    [~, leader2] = DxPx2d(data, param_est2);
    if type_exponent == 3
        [~, leader] = DxPx2d(data, param_est);
    end
end



for j = 1 : length(leader.value)
    leader_osc.value{j} = ...
        2 ^ (-j * rm1) * (leader2.value{j} ./ leader.value{j}) .^ (1 / ddelta);
    
    leader_osc.delta_gam   = delta_gam;
    leader_osc.delta_p = delta_p;
    leader_osc.type_exponent   = type_exponent;
end
