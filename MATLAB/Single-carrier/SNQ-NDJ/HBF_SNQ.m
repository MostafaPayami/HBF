%% HBF-ALS Hybrid Precoding

% clear
% clc

% load('Hybrid_Beamforming_mmWave_Massive_MIMO.mat');
% load('Ns=3.mat');

% Ns  = 6;
% NRF = 4;    % Number of RF chains 

SNR_dB = -30:5:10;
SNR = 10 .^ (SNR_dB / 10);
Nsample = size(H, 3);
smax = length(SNR);  

%% Water-Filling

% for n = 1:Nsample
%     sigma2 = diag(S(:, :, n)).^2;
%     sigma2 = sigma2(1:Ns).';
%     P = diag(sqrt(waterfill(Ns, 1 ./ sigma2 ./ SNR)));
%     Fopt(:, :, n) = Fopt(:, :, n) * P;
% end

%%

FRF = zeros(Nt,  NRF, Nsample);
FBB = zeros(NRF, Ns,  Nsample);
WRF = zeros(Nr,  NRF, Nsample);
WBB = zeros(NRF, Ns,  Nsample);

R_SNQ = zeros(smax, Nsample);
R_opt = zeros(smax, Nsample);

tic

% parfor n = 1:Nsample , iterF(n)
for n = 1:Nsample
    [FRF(:, :, n), FBB(:, :, n)] = HBF_NDJ_JP(Fopt(:, :, n), NRF);  
    FBB(:, :, n) = sqrt(Ns) * FBB(:, :, n) / norm(FRF(:, :, n) * FBB(:, :, n), 'fro');
    [WRF(:, :, n), WBB(:, :, n)] = HBF_NDJ_JP(Wopt(:, :, n), NRF); 

    for s = 1:smax
        R_SNQ(s, n) = log2(real(det(eye(Ns) + SNR(s)/Ns * pinv(WRF(:, :, n) * WBB(:, :, n)) * ...
                                    H(:, :, n) * FRF(:, :, n) * (FBB(:, :, n) * FBB(:, :, n)') * ...
                                    FRF(:, :, n)' * H(:, :, n)' * WRF(:, :, n) * WBB(:, :, n))));

        R_opt(s, n) = log2(real(det(eye(Ns) + SNR(s)/Ns * pinv(Wopt(:, :, n)) * H(:, :, n) * ... 
                                    Fopt(:, :, n) * Fopt(:, :, n)' * H(:, :, n)' * Wopt(:, :, n))));        
    end
end

CPU_time_SNQ = toc;
fprintf('\n   "CPU time for SNQ-NDJ method is %.4e seconds." \n \n', CPU_time_SNQ);

%% MMSE

% WBB_mmse = inv(WRF' * H  * FRF * FBB * FBB' * FRF' * H' * WRF + Ns / SNR * WRF' * WRF) * WRF' * H  * FRF * FBB;
% FBB_mmse = inv(FRF' * H' * WRF * WBB * WBB' * WRF' * H  * FRF + Ns / SNR * FRF' * FRF) * FRF' * H' * WRF * WBB;

%% Plot

figure
plot(SNR_dB, mean(R_opt, 2), 'b-o', 'LineWidth', 2)
grid on
hold on
plot(SNR_dB, mean(R_SNQ, 2), 'm--', 'LineWidth', 2)
% legend('Optimal Precoder', 'Proposed SNQ-NDJ')

%% Optimal Digital Precoder

    % Heff = (WRF(:, :, n))' * H(:, :, n) * FRF(:, :, n);
    % Gp = chol((FRF(:, :, n))' * FRF(:, :, n));
    % Gc = chol((WRF(:, :, n))' * WRF(:, :, n));
    % H2 = inv(Gc') * Heff * inv(Gp);
    % [Ub, Sb, Vb] = svd(H2);
    % FBB_new(:, :, n) = inv(Gp) * Vb(:, 1:Ns);
    % WBB_new(:, :, n) = inv(Gc) * Ub(:, 1:Ns);

    % for s = 1:smax
    %     R_HBF_new(s, n) = log2(det(eye(Ns) + SNR(s)/Ns * pinv(WRF(:, :, n) * WBB_new(:, :, n)) * ...
    %                       H(:, :, n) * FRF(:, :, n) * (FBB_new(:, :, n) * FBB_new(:, :, n)') * ...
    %                       FRF(:, :, n)' * H(:, :, n)' * WRF(:, :, n) * WBB_new(:, :, n)));
    % end