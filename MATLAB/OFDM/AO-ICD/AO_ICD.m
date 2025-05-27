%% Multicarrier AO-ICD Hybrid Beamforming Method

function [FRF, FBB, WRF, WBB] = AO_ICD(H, Ns, NRF, SNR)
[Nr, Nt, K] = size(H); 
P = Ns;                  % Total Power
sigma2 = P / SNR;        % Noise Variance
gamma2 = P / Nt / NRF;

% Analog Precoder 
F1  = mean(pagemtimes(H, 'ctranspose', H, 'none'), 3);
FRF = Algorithm_1(F1, NRF, gamma2 / sigma2);
    
% Digital Precoder
Heff = pagemtimes(H, FRF);
Q  = FRF' * FRF;
[~, ~, Ue] = pagesvd(pagemtimes(Heff, Q^(-0.5)), "econ");
Ue = Ue(:, 1:Ns, :);
Ge = eye(Ns);            % Equal Power
FBB = pagemtimes(Q^(-0.5), pagemtimes(Ue, Ge));    
FBB = sqrt(Ns) * FBB ./ pagenorm(pagemtimes(FRF, FBB), 'fro');   % Power Normalization

% Analog Combiner 
HF = pagemtimes(H, pagemtimes(FRF, FBB));
F2  = mean(pagemtimes(HF, 'none', HF, 'ctranspose'), 3);    
WRF = Algorithm_1(F2, NRF, 1 / Nr / sigma2);
    
% Digital Combiner
WHF = pagemtimes(WRF, 'ctranspose', HF, 'none');
J = pagemtimes(WHF, 'none', WHF, 'ctranspose') + (Ns / SNR) * (WRF' * WRF);
WBB = pagemldivide(J, WHF);
end

%% Algorithm 1 (Efficient Implementation)

function VRF = Algorithm_1(F, Nrf, a)
    [N, ~] = size(F);
    VRF = ones(N, Nrf);
    delta = 1;
    VRF0 = VRF;
    while delta > 1e-3
        for j = 1:Nrf
            Vj = VRF;
            Vj(:, j) = [];          
            C = eye(Nrf-1) + a * (Vj' * (F * Vj));
            G = a * F - a^2 * (F * ((Vj / C) * Vj') * F);
            eta = G * VRF(:, j) - diag(G) .* VRF(:, j);
            VRF(:, j) = exp(1i * angle(eta));
        end        
        delta = norm(VRF - VRF0, 'fro') / numel(VRF);
        VRF0 = VRF;
    end
end

%% Algorithm 1

% function VRF = Algorithm_1(F1, NRF, a)
%     [N, ~] = size(F1);
%     VRF = ones(N, NRF);
%     delta = 1;
%     VRF0 = VRF;
%     while delta > 1e-2
%         for j = 1:NRF
%             VRF_j = VRF;
%             VRF_j(:, j) = [];          
%             Cj = eye(NRF-1) + a * (VRF_j' * (F1 * VRF_j));
%             Gj = a * F1 - a^2 * (F1 * ((VRF_j / Cj) * VRF_j') * F1);
%             for i = 1:N
%                 eta_ij = sum(Gj(i, :) .* VRF(:, j)) - Gj(i, i) * VRF(i, j);
%                 if eta_ij == 0
%                     VRF(i, j) = 1;
%                 else
%                     VRF(i, j) = eta_ij / abs(eta_ij);
%                 end
%             end
%         end
%         delta = sum(abs(VRF0 - VRF), "all") / N / NRF;
%         VRF0 = VRF;
%     end
% end