%% Spatially Sparse Precoding (OMP)

% clear
clc

% load('Hybrid_Beamforming_mmWave_Massive_MIMO.mat');
% load('Ns=8, 2000.mat');

tic

Ns = 4;          % Number of data streams
NRF = 4;         % Number of RF chains 

SNR_dB = -35:5:5;             % Signal to Noise  ratio (dB)
SNR = 10 .^ (SNR_dB / 10);
realization = size(H, 3);
smax = length(SNR);           % Enable the parallel

% FRF_omp = zeros(Nt,  NRF, realization);
% FBB_omp = zeros(NRF, Ns,  realization);

WRF_omp_MMSE = zeros(Nr,  NRF, smax, realization);
WBB_omp_MMSE = zeros(NRF, Ns,  smax, realization);

R_omp_MMSE = zeros(smax, realization);
% R_opt = zeros(smax, realization);

for reali = 1:realization
    % [FRF_omp(:, :, reali), FBB_omp(:, :, reali)] = OMP(Fopt(:, :, reali), NRF, At(:, :, reali));
    % FBB_omp(:, :, reali) = sqrt(Ns) * FBB_omp(:, :, reali) / norm(FRF_omp(:, :, reali) * FBB_omp(:, :, reali), 'fro');

    for s = 1:smax
        [WRF_omp_MMSE(:, :, s, reali), WBB_omp_MMSE(:, :, s, reali)] = OMP_Combiner(Wmmse(:, :, s, reali), NRF, Ar(:, :, reali), Eyy(:, :, s, reali));
        R_omp_MMSE(s, reali) = log2(det(eye(Ns) + SNR(s)/Ns * pinv(WRF_omp_MMSE(:, :, s, reali) * WBB_omp_MMSE(:, :, s, reali)) * ...
                          H(:, :, reali) * FRF_omp(:, :, reali) * (FBB_omp(:, :, reali) * FBB_omp(:, :, reali)') * ...
                          FRF_omp(:, :, reali)' * H(:, :, reali)' * WRF_omp_MMSE(:, :, s, reali) * WBB_omp_MMSE(:, :, s, reali)));
        
        % R_opt(s, reali) = log2(det(eye(Ns) + SNR(s)/Ns * pinv(Wopt(:, :, reali)) * H(:, :, reali) * ... 
        %                   Fopt(:, :, reali) * Fopt(:, :, reali)' * H(:, :, reali)' * Wopt(:, :, reali)));
    end
end

toc

%% Plot

figure
plot(SNR_dB, sum(R_omp_MMSE, 2)/realization, 'Marker', '^', 'LineWidth', 1.5, 'Color', [0 0.5 0])
grid on
hold on
plot(SNR_dB, sum(R_opt, 2)/realization, 'r-o', 'LineWidth', 1.5)