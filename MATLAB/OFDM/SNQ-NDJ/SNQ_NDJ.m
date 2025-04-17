%% Hybrid Beamforming Joint HBF-JLS-NDJ Algorithm for OFDM Systems (Joint-LS and Newton-DJ method) 

%% Based on Tensor Computations (Product)

function [FRF, FBB] = SNQ_NDJ(Fopt, NRF)

alpha = 0.5;
Nmax  = 100;
epsilon = 1e-4;

%% NRF = NS

[Nt, Ns, K] = size(Fopt);
I_NRF = eye(NRF); 

% In = repmat(eye(NRF), [1, 1, K]);

%% Optimal Initialization: FBB, T and FRF=exp(1i * T)

% R       = abs(Fopt);                   % Optimal Analog and Digital Precoder for NRF=2*Ns
% rho     = max(R, [], 1);
% T0      = angle(Fopt);
% Tp      = T0 + acos(R ./ rho);
% Tn      = T0 - acos(R ./ rho);
% 
% Nk  = NRF - Ns;
% T   = [Tp(:, 1:Nk) , Tn(:, 1:Nk) , T0(:, Nk+1:end)];

% % T0_2Ns  = [Tp , Tn];
% % FBB_2Ns = [eye(Ns) ; eye(Ns)] / 2;
% 
% % T0_Ns  = angle(Fopt);                  % Initial Analog and Digital Precoder for NRF=Ns
% % FBB_Ns = eye(Ns);
% 
% % FBB = [FBB_Ns(1:Nk, :) / 2 ; FBB_Ns(1:Nk, :) / 2 ; FBB_Ns(Nk+1:end, :)];
% % FRF = exp(1i * T);
 
% % Fopt = Fopt ./ rho;

%% Constant Initialization

% FBB = eye(NRF, Ns);
% T = zeros(Nt, NRF);
% FBB_2Ns = [eye(Ns) ; eye(Ns)];
% FBB = FBB_2Ns(1:NRF, 1:Ns);

%% Random Initialization

% T   = (2 * rand(Nt, NRF) - 1) * pi;
% FRF = exp(1i * T) / sqrt(Nt);
% FBB = pinv(FRF) * Fopt;

%% Random Initialization

T   = unifrnd(-pi, pi, [Nt, NRF]);        % T   = 2 * pi * rand(Nt, NRF) - pi;      
Psi = unifrnd(-pi, pi, [NRF, Ns, K]);     % Psi = 2 * pi * rand(NRF, Ns, K) - pi;
R   = repmat(eye(NRF, Ns), [1, 1, K]);    % R   = random('Rayleigh', 1, [NRF, Ns, K]);

FRF = exp(1i * T) / sqrt(Nt);
FBB = R .* exp(1i * Psi);

% g0  = norm(Fopt - FRF * FBB, 'fro')^2;
g0  = sum(pagenorm(Fopt - pagemtimes(FRF, FBB), 'fro').^2, "all");

%% Joint Analog and Digital Precoder Design with Diagonal Jacobian Newton Method (SNQ-NDJ) (JP-NDJ)

for k = 1:Nmax
    % Analog Precoder
    Bm_eff = sum(pagemtimes(FBB,  'none', FBB, 'ctranspose'), 3) .* (1 - I_NRF);   
    Z      = sum(pagemtimes(Fopt, 'none', FBB, 'ctranspose'), 3) - FRF * Bm_eff;
    Phi    = angle(Z);
    T      = T + sin(Phi - T);                
    FRF    = exp(1i * T) / sqrt(Nt);

    % Digital Precoder    
    Dm  = (FRF' * FRF) .* (1 - I_NRF);        % Dm = FRF' * FRF - In; 
    W   = pagemtimes(FRF, 'ctranspose', Fopt, 'none') - pagemtimes(Dm, FBB);    
    Omg = angle(W);
    Psi = Psi + sin(Omg - Psi);  
    R   = (1 - alpha) * R + alpha * abs(W);        % Relaxed Fixed-Point Iteration    
    FBB = R .* exp(1i * Psi);

    % Convergence (Criterion)
    g  = sum(pagenorm(Fopt - pagemtimes(FRF, FBB), 'fro').^2, "all");
    if abs(g - g0) / (g0 + eps) < epsilon    
        break;
    end    
    g0 = g;

    % g  = sum(abs(Phi - T), "all") / (Nt*NRF) + sum(abs(Psi - Omg), "all") / (NRF*Ns*K) + sum(abs(FRF' * Fopt - Dm * FBB), "all") / (NRF*Ns*K);
    % if g < epsilon    
    % if g / (Nt*NRF + 2*NRF*Ns*K) < epsilon    
    %     break;
    % end 
end

end
