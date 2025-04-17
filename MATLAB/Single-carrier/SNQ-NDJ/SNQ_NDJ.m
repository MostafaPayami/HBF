%% SNQ-NDJ Hybrid Beamforming Method 
function [FRF, FBB] = SNQ_NDJ(Fopt, NRF)

alpha = 0.5;
Nmax  = 100;
epsilon = 1e-4;
I_NRF = eye(NRF);
[Nt, Ns] = size(Fopt);

%% Initialization
T   = unifrnd(-pi, pi, [Nt, NRF]);
Psi = unifrnd(-pi, pi, [NRF, Ns]);
R   = eye(NRF, Ns);                   

%% Hybrid Precoders Optimization   
FRF = exp(1i * T) / sqrt(Nt);
FBB = R .* exp(1i * Psi);
g0  = norm(Fopt - FRF * FBB, 'fro')^2;

for k = 1:Nmax
    % Analog Precoder
    Bm  = (FBB * FBB') .* (1 - I_NRF);            
    Phi = angle(Fopt * FBB' - FRF * Bm);
    T   = T + sin(Phi - T);                
    FRF = exp(1i * T) / sqrt(Nt);

    % Digital Precoder
    Dm  = (FRF' * FRF) .* (1 - I_NRF); 
    W   = FRF' * Fopt - Dm * FBB;
    Omg = angle(W);
    Psi = Psi + sin(Omg - Psi);  
    R   = (1 - alpha) * R + alpha * abs(W); 
    FBB = R .* exp(1i * Psi);

    % Convergence Criterion
    g  = norm(Fopt - FRF * FBB, 'fro')^2;        
    if abs(g - g0) / (g0 + eps) < epsilon    
        break;
    end    
    g0 = g;
end
end

%% Unitary Baseband Precoder
% [U, ~, V] = svd(W, "econ");
% FBB = U * V';
