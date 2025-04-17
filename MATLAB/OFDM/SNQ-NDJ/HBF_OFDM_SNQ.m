%% Multicarrier SNQ-NDJ Hybrid Beamforming Method 

% clear
% clc

% Nt = 256;       % Number of transmit antennas
% Nr = 64;        % Number of receive antennas
% K  = 128;       % Number of OFDM subcarriers
% % Ns  = 5;        % Number of data streams
% % NRF = 5;        % Number of RF chains
% Nc = 6;                               % Number of clusters
% Nray = 12;                            % Number of rays in each cluster
% angle_sigma = 15 / 180 * pi;          % Standard deviation of Tx/Rx azimuth and elevation angles
% gamma = sqrt((Nt*Nr) / (Nc*Nray));    % Normalization factor
% sigma = 1;                            % Normalization condition of the H

Nsample = size(H, 4);
SNRdB = -30:5:10;
SNR   = 10 .^ (SNRdB / 10);
SE_SNQ = zeros(length(SNR), Nsample); 
SE_opt = zeros(length(SNR), Nsample); 

%% SNQ-NDJ Method

tic
for n = 1:Nsample
    [FRF, FBB] = HBF_NDJ_OFDM(Fopt(:, :, :, n), NRF);
    FBB = sqrt(Ns) * FBB ./ pagenorm(pagemtimes(FRF, FBB), 'fro');     
    [WRF, WBB] = HBF_NDJ_OFDM(Wopt(:, :, :, n), NRF);

    % Spectral Efficiency
    % for k = 1:K
    %     for s = 1:length(SNR) 
    %         SE_opt(s, n) = SE_opt(s, n) + log2(real(det(eye(Ns) + SNR(s) / Ns * pinv(Wopt(:,:,k, n)) * H(:,:,k, n) * ...
    %                                           (Fopt(:,:,k, n) * Fopt(:,:,k, n)') * H(:,:,k, n)' * Wopt(:,:,k, n)))) / K;            
    %         SE_SNQ(s, n) = SE_SNQ(s, n) + log2(real(det(eye(Ns) + SNR(s) / Ns * pinv(WRF * WBB(:,:,k)) * H(:,:,k, n) * FRF * ...
    %                                           (FBB(:,:,k) * FBB(:,:,k)') * FRF' * H(:,:,k, n)' * WRF * WBB(:,:,k)))) / K;
    %     end
    % end

    % % Spectral Efficiency
    % Ft = pagemtimes(FRF, FBB);
    % Wr = pagemtimes(WRF, WBB);
    % WHF  = pagemtimes(Wr, 'ctranspose', pagemtimes(H(:, :, :, n), Ft), 'none');
    % WHF2 = pagemtimes(WHF, 'none', WHF, 'ctranspose');
    % WHF3 = pagemtimes(pageinv(pagemtimes(Wr, 'ctranspose', Wr, 'none')), WHF2);
    % Eig  = pageeig(WHF3);
    % for s = 1:length(SNR) 
    %     SE_SNQ(s, n) = sum(log2(real(1 + SNR(s) / Ns * Eig)), "all") / K;
    % end

end
CPU_time_SNQ = toc;
fprintf('  "CPU time for Multicarrier SNQ-NDJ method is %f seconds." \n', CPU_time_SNQ);

%% Plot 

% figure
% plot(SNRdB, mean(SE_opt, 2), 'b-o', 'LineWidth', 3, 'DisplayName', 'Optimal Precoder')
% grid on
% hold on
% plot(SNRdB, mean(SE_SNQ, 2), 'm-p', 'LineWidth', 3, 'DisplayName', 'NSQ-NDJ')
% xlabel('SNR (dB)')
% ylabel('Spectral Efficiency (bits/sec/Hz)')
% legend('show')
