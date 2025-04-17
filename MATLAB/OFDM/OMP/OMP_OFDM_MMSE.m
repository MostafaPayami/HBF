%% Multicarrier OMP Hybrid Beamforming Method 

% clear
% clc

% load('Hybrid_Beamforming_mmWave_Massive_MIMO.mat');
% load('Ns=3.mat');

Nt = 256;       % Number of transmit antennas
Nr = 64;        % Number of receive antennas
K  = 128;       % Number of OFDM subcarriers
% Ns  = 5;        % Number of data streams
% NRF = 5;        % Number of RF chains
Nc = 6;                               % Number of clusters
Nray = 12;                            % Number of rays in each cluster
angle_sigma = 15 / 180 * pi;          % Tx/Rx azimuth and elevation anglular spread 
gamma = sqrt((Nt*Nr) / (Nc*Nray));    % Normalization factor
sigma = 1;                            % Normalization condition of the H

Nsample = size(H, 4);
SNRdB  = -30:5:10;
SNR    = 10 .^ (SNRdB / 10);
SE_OMP = zeros(length(SNR), Nsample); 

%% OMP Method

tic
for n = 1:Nsample
    [FRF, FBB] = OMP(Fopt(:, :, :, n), NRF, At(:, :, n));

    [FRF, FBB, WRF, WBB] = AO_ICD(H(:, :, :, n), Ns, NRF, SNR(s));

    FBB = sqrt(Ns) * FBB ./ pagenorm(pagemtimes(FRF, FBB), 'fro');
    [WRF, WBB] = OMP(Wopt(:, :, :, n), NRF, Ar(:, :, n));

    % Spectral Efficiency
    for k = 1:K
        for s = 1:length(SNR) 
            SE_OMP(s, n) = SE_OMP(s, n) + log2(real(det(eye(Ns) + SNR(s) / Ns * pinv(WRF * WBB(:,:,k)) * H(:,:,k, n) * FRF * ...
                                               (FBB(:,:,k) * FBB(:,:,k)') * FRF' * H(:,:,k, n)' * WRF * WBB(:,:,k)))) / K;
        end
    end
end
CPU_time_OMP = toc;
fprintf('  "CPU time for Multicarrier OMP method is %f seconds." \n', CPU_time_OMP);

%% Plot

% figure
% grid on
% hold on
plot(SNRdB, mean(SE_OMP, 2), 'g-*', 'LineWidth', 3, 'DisplayName', 'OMP')
% xlabel('SNR (dB)')
% ylabel('Spectral Efficiency (bits/s/Hz)')
