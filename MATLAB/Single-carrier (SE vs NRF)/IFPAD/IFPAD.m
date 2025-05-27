%% IFPAD Hybrid Beamforming Method

function [FRF, FBB] = IFPAD(Fopt, NRF)

alpha = 0.75;
Nmax  = 100;
epsilon = 1e-4;
[Nt, Ns] = size(Fopt);

%% Random Initialization

% T   = 2 * pi * rand(Nt, NRF) - pi; 
% Psi = 2 * pi * rand(NRF, Ns) - pi; 
% R   = eye(NRF, Ns); 

%% Optimal Initialization: based on Optimal Hybrid Precoders for the case of NRF = 2 * Ns [1]
% [1] E. Zhang and C. Huang, "On Achieving Optimal Rate of Digital Precoder by RF-Baseband Codesign for MIMO Systems," in 2014 IEEE 80th Vehic. Tech. Conf. (VTC 2014-Fall), Sep. 2014, pp. 1–5.

Ropt  = abs(Fopt);      
r_max = max(Ropt, [], 1);
Tp    = angle(Fopt) + acos(Ropt ./ r_max);
Tn    = angle(Fopt) - acos(Ropt ./ r_max);
R0    = eye(Ns) .* r_max * sqrt(Nt);

T   = [Tp(:, 1:NRF-Ns) , Tn(:, 1:NRF-Ns) , unifrnd(-pi, pi, [Nt, 2*Ns-NRF])];
R   = blkdiag([R0(1:NRF-Ns, 1:NRF-Ns) ; R0(1:NRF-Ns, 1:NRF-Ns)] / 2, eye(2*Ns-NRF));
Phi = zeros(NRF, Ns);

%% IFPAD Method   
FRF = exp(1i * T) / sqrt(Nt);
FBB = R .* exp(1i * Phi);
g0  = norm(Fopt - FRF * FBB, 'fro')^2;

for k = 1:Nmax
    % Analog Precoder
    LBB = (FBB * FBB') .* (1 - eye(NRF));            
    Psi = angle(Fopt * FBB' - FRF * LBB);
    T   = T + sin(Psi - T);                
    FRF = exp(1i * T) / sqrt(Nt);

    % Digital Precoder
    LRF = (FRF' * FRF) .* (1 - eye(NRF)); 
    W   = FRF' * Fopt - LRF * FBB;
    Omg = angle(W);
    Phi = Phi + sin(Omg - Phi);  
    R   = (1 - alpha) * R + alpha * abs(W); 
    FBB = R .* exp(1i * Phi);

    % Convergence Criterion
    g  = norm(Fopt - FRF * FBB, 'fro')^2;        
    if abs(g - g0) / (g0 + eps) < epsilon    
        break;
    end    
    g0 = g;
end
end