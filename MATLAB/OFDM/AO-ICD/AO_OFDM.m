%% Multicarrier AO-ICD Hybrid Beamforming Method

% clear
% clc

% load('Hybrid_Beamforming_mmWave_Massive_MIMO.mat');
% load('Ns=8, 2000.mat');

% Nt = 256;       % Number of transmit antennas
% Nr = 64;        % Number of receive antennas
% K  = 128;       % Number of OFDM subcarriers
% % Ns  = 5;        % Number of data streams
% % NRF = 7;        % Number of RF chains
% SNRdB = -30:5:10;
% SNR   = 10 .^ (SNRdB / 10);
% Nsample = size(H, 4);

SE_AO = zeros(length(SNR), Nsample); 

SNR = 1; 
%% AO-ICD Method (2017)

tic
for n = 1:Nsample
    % n
    % for s = 1:length(SNR) 
       [FRF, FBB, WRF, WBB] = AO_ICD(H(:, :, :, n), Ns, NRF, SNR);
       % [FRF_ao, FBB_ao, WRF_ao, WBB_ao, iterF(n), iterW(n)] = AO_ICD(H(:, :, n), Ns, NRF, SNR);
       
       % Spectral Efficiency
       % for k = 1:K
       %     SE_AO(s, n) = SE_AO(s, n) + log2(real(det(eye(Ns) + SNR(s) / Ns * pinv(WRF * WBB(:,:,k)) * H(:,:,k, n) * FRF * ... 
       %                                       (FBB(:,:,k) * FBB(:,:,k)') * FRF' * H(:,:,k, n)' * WRF * WBB(:,:,k)))) / K;
       % end
    % end
end
CPU_time_AO = toc;
fprintf('  "CPU time for Multicarrier AO-ICD method is %f seconds." \n', CPU_time_AO);

%% Plot

% figure
% grid on
% hold on
plot(SNRdB, mean(SE_AO, 2), 'c-.*', 'LineWidth', 4, 'DisplayName', 'AO-ICD')
% xlabel('SNR (dB)')
% ylabel('Spectral Efficiency (bits/s/Hz)')

