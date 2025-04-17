%% MO-AltMin Hybrid Precoding

% clear
% clc

addpath(pwd);
cd manopt;
addpath(genpath(pwd));
cd ..;

% load('Hybrid_Beamforming_mmWave_Massive_MIMO.mat');
% load('Ns=3.mat');

tic

% Ns  = 4;
NRF = 4;

SNRdB = -35:5:10;
SNR    = 10 .^ (SNRdB / 10);
Nsample = size(H, 3);
smax = length(SNR);         

FRF_mo = zeros(Nt,  NRF, Nsample);
FBB_mo = zeros(NRF, Ns,  Nsample);

WRF_mo = zeros(Nr,  NRF, Nsample);
WBB_mo = zeros(NRF, Ns,  Nsample);

R_MO = zeros(smax, Nsample);

% MSE_mo = zeros(Nsample, 1);
% CS_mo = zeros(Nsample, 1);

% parfor n = 1:Nsample
for n = 1:Nsample
    n
    [FRF_mo(:, :, n), FBB_mo(:, :, n)] = MO_AltMin(Fopt(:, :, n), NRF); % , Fopt_2Ns , HBF_phase_OMP(:, :, reali)
    FBB_mo(:, :, n) = sqrt(Ns) * FBB_mo(:, :, n) / norm(FRF_mo(:, :, n) * FBB_mo(:, :, n), 'fro');
    % MSE_mo(reali) = sum(abs(Fopt(:, :, reali) - FRF_mo(:, :, reali)*FBB_mo(:, :, reali)).^2, "all");
    % CS_mo(reali)  = sum(conj(Fopt(:, :, reali)) .* (FRF_mo(:, :, reali) * FBB_mo(:, :, reali)), "all") / Ns;
    [WRF_mo(:, :, n), WBB_mo(:, :, n)] = MO_AltMin(Wopt(:, :, n), NRF); % , Wopt_2Ns

    % for s = 1:smax
    %     R_MO(s, n) = log2(det(eye(Ns) + SNR(s)/Ns * pinv(WRF_mo(:, :, n) * WBB_mo(:, :, n)) * ...
    %                      H(:, :, n) * FRF_mo(:, :, n) * (FBB_mo(:, :, n) * FBB_mo(:, :, n)') * ...
    %                      FRF_mo(:, :, n)' * H(:, :, n)' * WRF_mo(:, :, n) * WBB_mo(:, :, n)));
    % end
end

CPU_time_MO = toc;
fprintf('\n   "CPU time for MO-AltMin method is %.4e seconds." \n \n', CPU_time_MO);

for n = 1:Nsample
    for s = 1:smax
        R_MO(s, n) = log2(det(eye(Ns) + SNR(s)/Ns * pinv(WRF_mo(:, :, n) * WBB_mo(:, :, n)) * ...
                         H(:, :, n) * FRF_mo(:, :, n) * (FBB_mo(:, :, n) * FBB_mo(:, :, n)') * ...
                         FRF_mo(:, :, n)' * H(:, :, n)' * WRF_mo(:, :, n) * WBB_mo(:, :, n)));
    end
end

%% Plot

% figure
plot(SNRdB, mean(R_MO, 2), 'r:', 'LineWidth', 2)
% grid on
% hold on