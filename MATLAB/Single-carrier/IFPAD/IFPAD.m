%% IFPAD Hybrid Beamforming Method

function [FRF, FBB] = IFPAD(Fopt, NRF)

alpha = 0.5;
Nmax  = 100;
epsilon = 1e-4;
[Nt, Ns] = size(Fopt);

%% Initialization
T   = unifrnd(-pi, pi, [Nt, NRF]);
Phi = unifrnd(-pi, pi, [NRF, Ns]);
R   = eye(NRF, Ns);                   

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

% % Unitary Digital Precoder
% [U, ~, V] = svd(FRF' * Fopt - LRF * FBB, "econ");
% FBB = U * V';
