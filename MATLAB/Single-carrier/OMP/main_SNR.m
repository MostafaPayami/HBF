%% Spatially Sparse Precoding (OMP)

% clear
% clc

% load('Hybrid_Beamforming_mmWave_Massive_MIMO.mat');
% load('Ns=8, 2000.mat');

% Ns  = 4;         % Number of data streams
NRF = 4;         % Number of RF chains 

SNRdB = -35:5:10;             % Signal to Noise  ratio (dB)
SNR   = 10 .^ (SNRdB / 10);
Nsample = size(H, 3);
smax = length(SNR);           % Enable the parallel

FRF_omp = zeros(Nt,  NRF, Nsample);
FBB_omp = zeros(NRF, Ns,  Nsample);

WRF_omp = zeros(Nr,  NRF, Nsample);
WBB_omp = zeros(NRF, Ns,  Nsample);

AoD_indx_omp = zeros(NRF, Nsample);
AoA_indx_omp = zeros(NRF, Nsample);

R_OMP = zeros(smax, Nsample);

tic

for n = 1:Nsample
    [FRF_omp(:, :, n), FBB_omp(:, :, n), AoD_indx_omp(:, n)] = OMP(Fopt(:, :, n), NRF, At(:, :, n));
    FBB_omp(:, :, n) = sqrt(Ns) * FBB_omp(:, :, n) / norm(FRF_omp(:, :, n) * FBB_omp(:, :, n), 'fro');
    [WRF_omp(:, :, n), WBB_omp(:, :, n), AoA_indx_omp(:, n)] = OMP(Wopt(:, :, n), NRF, Ar(:, :, n));

    for s = 1:smax
        R_OMP(s, n) = log2(det(eye(Ns) + SNR(s)/Ns * pinv(WRF_omp(:, :, n) * WBB_omp(:, :, n)) * ...
                          H(:, :, n) * FRF_omp(:, :, n) * (FBB_omp(:, :, n) * FBB_omp(:, :, n)') * ...
                          FRF_omp(:, :, n)' * H(:, :, n)' * WRF_omp(:, :, n) * WBB_omp(:, :, n)));
    end
end

CPU_time_OMP = toc;
fprintf('\n   "CPU time for OMP method is %.4e seconds." \n \n', CPU_time_OMP);

%% Plot

% figure
plot(SNRdB, mean(R_OMP, 2), 'g-*', 'LineWidth', 2)
% grid on
% hold on
