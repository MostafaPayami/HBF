%% Hybrid Beamforming for Millimeter-Wave Massive MIMO-OFDM Wireless Systems 

addpath(genpath(pwd));
rng('shuffle') 
% clear all
% clc

%% Massive MIMO-OFDM System Parameters

Nt  = 256;     % Number of transmit antennas
Nr  = 64;      % Number of receive antennas
K   = 128;     % Number of OFDM subcarriers
Ns  = 5;       % Number of data streams
NRF = 7;       % Number of RF chains

SNRdB = -30:5:10;          % Signal-to-noise ratio (dB)
SNR   = 10.^(SNRdB / 10);  % Signal-to-noise ratio
Nsamples = 20;             % Number of channel realizations

%% mmWave OFDM Channel Realizations

tic
[Fopt, Wopt, H, At, Ar] = Channel_Generation(Nt, Nr, K, Ns, Nsamples); 
CPU_time_Channel = toc;
fprintf('   "%d OFDM channel realizations were generated." \n', Nsamples);

%% Multicarrier IFPAD Hybrid Beamforming Method

SE_IFPAD = zeros(length(SNR), Nsamples);
SE_opt   = zeros(length(SNR), Nsamples);

tic
for n = 1:Nsamples
    [FRF, FBB] = IFPAD(Fopt(:, :, :, n), NRF);  
    [WRF, WBB] = IFPAD(Wopt(:, :, :, n), NRF); 
    FBB = sqrt(Ns) * FBB ./ pagenorm(pagemtimes(FRF, FBB), 'fro');     

    % Spectral Efficiency
    for k = 1:K
        for s = 1:length(SNR) 
            SE_IFPAD(s, n) = SE_IFPAD(s, n) + log2(real(det(eye(Ns) + SNR(s) / Ns * pinv(WRF * WBB(:,:,k)) * H(:,:,k, n) * (FRF * ...
                                                           (FBB(:,:,k) * FBB(:,:,k)') * FRF') * H(:,:,k, n)' * WRF * WBB(:,:,k)))) / K;
            SE_opt(s, n)   = SE_opt(s, n)   + log2(real(det(eye(Ns) + SNR(s) / Ns * pinv(Wopt(:,:,k, n)) * H(:,:,k, n) * ...
                                                           (Fopt(:,:,k, n) * Fopt(:,:,k, n)') * H(:,:,k, n)' * Wopt(:,:,k, n)))) / K;            
        end
    end
end
CPU_time_IFPAD = toc;     % For accurate CPU time comment out the Spectral Efficiency calculation step.
fprintf('   "CPU time for the IFPAD method is %f ms." \n', CPU_time_IFPAD / Nsamples * 1000);

%% Multicarrier OMP Hybrid Beamforming Method

SE_OMP = zeros(length(SNR), Nsamples);

tic
for n = 1:Nsamples
    [FRF, FBB] = OMP(Fopt(:, :, :, n), NRF, At(:, :, n));
    [WRF, WBB] = OMP(Wopt(:, :, :, n), NRF, Ar(:, :, n));
    FBB = sqrt(Ns) * FBB ./ pagenorm(pagemtimes(FRF, FBB), 'fro');

    % Spectral Efficiency
    for k = 1:K
        for s = 1:length(SNR) 
            SE_OMP(s, n) = SE_OMP(s, n) + log2(real(det(eye(Ns) + SNR(s) / Ns * pinv(WRF * WBB(:,:,k)) * H(:,:,k, n) * (FRF * ...
                                                       (FBB(:,:,k) * FBB(:,:,k)') * FRF') * H(:,:,k, n)' * WRF * WBB(:,:,k)))) / K;
        end
    end
end
CPU_time_OMP = toc;
fprintf('   "CPU time for the OMP method is %f ms." \n', CPU_time_OMP / Nsamples * 1000);

%% Multicarrier AO-ICD Hybrid Beamforming Method

SE_AO = zeros(length(SNR), Nsamples);

tic
for n = 1:Nsamples
    for s = 1:length(SNR)
        [FRF, FBB, WRF, WBB] = AO_ICD(H(:, :, :, n), Ns, NRF, SNR(s));

        % Spectral Efficiency
        for k = 1:K
            SE_AO(s, n) = SE_AO(s, n) + log2(real(det(eye(Ns) + SNR(s) / Ns * pinv(WRF * WBB(:,:,k)) * H(:,:,k, n) * (FRF * ... 
                                                     (FBB(:,:,k) * FBB(:,:,k)') * FRF') * H(:,:,k, n)' * WRF * WBB(:,:,k)))) / K;
        end
    end
end
CPU_time_AO = toc;
fprintf('   "CPU time for the AO-ICD method is %f ms." \n', CPU_time_AO / Nsamples / length(SNR) * 1000);

%% Multicarrier MO-AltMin Hybrid Beamforming Method

SE_MO = zeros(length(SNR), Nsamples);

tic
for n = 1:Nsamples
    [FRF, FBB] = MO_AltMin(Fopt(:, :, :, n), NRF);
    [WRF, WBB] = MO_AltMin(Wopt(:, :, :, n), NRF);
    FBB = sqrt(Ns) * FBB ./ pagenorm(pagemtimes(FRF, FBB), 'fro');   

    % Spectral Efficiency
    for k = 1:K
        for s = 1:length(SNR) 
            SE_MO(s, n) = SE_MO(s, n) + log2(real(det(eye(Ns) + SNR(s) / Ns * pinv(WRF * WBB(:,:,k)) * (H(:,:,k, n) * (FRF * ...
                                                     (FBB(:,:,k) * FBB(:,:,k)') * FRF') * H(:,:,k, n)') * WRF * WBB(:,:,k)))) / K;
        end
    end
end
CPU_time_MO = toc;
fprintf('   "CPU time for the MO-AltMin method is %f ms." \n', CPU_time_MO / Nsamples * 1000);

%% Performance Evaluation 

figure
grid on
hold on
plot(SNRdB, mean(SE_opt, 2),   'b-o',  'LineWidth', 2.5)
plot(SNRdB, mean(SE_IFPAD, 2), 'm-p',  'LineWidth', 2.5)
plot(SNRdB, mean(SE_MO, 2),    'k:+',  'LineWidth', 2.5)
plot(SNRdB, mean(SE_AO, 2),    'c-.*', 'LineWidth', 2.5)
plot(SNRdB, mean(SE_OMP, 2),   'g-s',  'LineWidth', 2.5)
legend('Optimal Precoder', 'IFPAD', 'MO-AltMin', 'AO-ICD', 'OMP', 'Location', 'southeast')
xlabel('SNR (dB)')
ylabel('Spectral Efficiency (bits/s/Hz)')
axis([-30 10 0 60])
