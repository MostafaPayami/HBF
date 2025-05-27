%% Multicarrier IFPAD Hybrid Beamforming Method

function [FRF, FBB] = IFPAD(Fopt, NRF)

alpha = 0.5;
Nmax  = 100;
epsilon = 1e-4;
[Nt, Ns, K] = size(Fopt);

%% Initialization

T   = unifrnd(-pi, pi, [Nt, NRF]);      
Psi = unifrnd(-pi, pi, [NRF, Ns, K]);  
R   = repmat(eye(NRF, Ns), [1, 1, K]); 

%% IFPAD Method  

FRF = exp(1i * T) / sqrt(Nt);
FBB = R .* exp(1i * Psi);
g0  = sum(pagenorm(Fopt - pagemtimes(FRF, FBB), 'fro').^2, "all");

for k = 1:Nmax
    % Analog Precoder
    LBB = sum(pagemtimes(FBB,  'none', FBB, 'ctranspose'), 3) .* (1 - eye(NRF));   
    Z   = sum(pagemtimes(Fopt, 'none', FBB, 'ctranspose'), 3) - FRF * LBB;
    Phi = angle(Z);
    T   = T + sin(Phi - T);                
    FRF = exp(1i * T) / sqrt(Nt);

    % Digital Precoder    
    LRF = (FRF' * FRF) .* (1 - eye(NRF)); 
    W   = pagemtimes(FRF, 'ctranspose', Fopt, 'none') - pagemtimes(LRF, FBB);    
    Omg = angle(W);
    Psi = Psi + sin(Omg - Psi);  
    R   = (1 - alpha) * R + alpha * abs(W);      
    FBB = R .* exp(1i * Psi);

    % Convergence Criterion
    g  = sum(pagenorm(Fopt - pagemtimes(FRF, FBB), 'fro').^2, "all");
    if abs(g - g0) / (g0 + eps) < epsilon    
        break;
    end    
    g0 = g; 
end
end