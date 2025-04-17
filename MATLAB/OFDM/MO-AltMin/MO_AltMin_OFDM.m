%% Multicarrier MO-AltMin Hybrid Beamforming Method

% clear
% clc

addpath(pwd);
cd manopt;
addpath(genpath(pwd));
cd ..;

% load('Hybrid_Beamforming_mmWave_Massive_MIMO.mat');
% load('Ns=3.mat');

% Nt = 256;       % Number of transmit antennas
% Nr = 64;        % Number of receive antennas
% K  = 128;       % Number of OFDM subcarriers
% % Ns  = 5;        % Number of data streams
% % NRF = 7;        % Number of RF chains
% Nc = 6;                               % Number of clusters
% Nray = 12;                            % Number of rays in each cluster
% angle_sigma = 15 / 180 * pi;          % Standard deviation of Tx/Rx azimuth and elevation angles
% gamma = sqrt((Nt*Nr) / (Nc*Nray));    % Normalization factor
% sigma = 1;                            % Normalization condition of the H

Nsample = size(H, 4);
SNRdB = -30:5:10;
SNR   = 10 .^ (SNRdB / 10);
SE_MO = zeros(length(SNR), Nsample); 

%% MO-AltMin

tic
for n = 1:Nsample
    n
    [FRF, FBB] = MO_AltMin(Fopt(:, :, :, n), NRF);
    FBB = sqrt(Ns) * FBB ./ pagenorm(pagemtimes(FRF, FBB), 'fro');   
    [WRF, WBB] = MO_AltMin(Wopt(:, :, :, n), NRF);
    % Spectral Efficiency
    % for k = 1:K
    %     for s = 1:length(SNR) 
    %         SE_MO(s, n)  = SE_MO(s, n)  + log2(real(det(eye(Ns) + SNR(s) / Ns * pinv(WRF * WBB(:,:,k)) * (H(:,:,k, n) * ...
    %                                            (FRF * (FBB(:,:,k) * FBB(:,:,k)') * FRF') * H(:,:,k, n)') * WRF * WBB(:,:,k)))) / K;
    %     end
    % end
end
CPU_time_MO = toc;
fprintf('  "CPU time for Multicarrier MO-AltMin method is %f seconds." \n', CPU_time_MO);

%% Plot 

% figure
% grid on
% hold on
plot(SNRdB, mean(SE_MO, 2), 'k:+', 'LineWidth', 4, 'DisplayName', 'MO-AltMin')
% xlabel('SNR (dB)')
% ylabel('Spectral Efficiency (bits/s/Hz)')
