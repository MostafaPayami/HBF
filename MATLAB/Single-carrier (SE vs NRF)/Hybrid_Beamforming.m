%% Hybrid Beamforming for Millimeter-Wave Massive MIMO Wireless Systems 

addpath(genpath(pwd));
rng('shuffle') 
% clear all
% clc

%% Massive MIMO System Parameters

Nt  = 256;     % Number of transmit antennas
Nr  = 64;      % Number of receive antennas
Ns  = 5;       % Number of data streams
NRF = Ns:2*Ns; % Number of RF chains 

SNRdB = 0;                 % Signal-to-noise ratio (dB)
SNR   = 10.^(SNRdB / 10);  % Signal-to-noise ratio
Nsamples = 100;            % Number of channel realizations

%% mmWave Channel Realizations

tic
[Fopt, Wopt, H, At, Ar] = Channel_Generation(Nt, Nr, Ns, Nsamples); 
CPU_time_Channel = toc;
fprintf('   "%d random channel realizations were generated in %.4f seconds." \n', Nsamples, CPU_time_Channel);

%% IFPAD Hybrid Beamforming Method

SE_IFPAD = zeros(length(NRF), Nsamples);
SE_opt   = zeros(length(NRF), Nsamples);

tic
for n = 1:Nsamples
   for s = 1:length(NRF)
       [FRF, FBB] = IFPAD(Fopt(:, :, n), NRF(s));
       [WRF, WBB] = IFPAD(Wopt(:, :, n), NRF(s)); 
       FBB = sqrt(Ns) / norm(FRF * FBB, 'fro') * FBB;

        % Spectral Efficiency
        SE_IFPAD(s, n) = log2(real(det(eye(Ns) + SNR / Ns * pinv(WRF * WBB) * H(:, :, n) * ... 
                                      (FRF * (FBB * FBB') *  FRF') * H(:, :, n)' * WRF * WBB)));
        SE_opt(s, n)   = log2(real(det(eye(Ns) + SNR / Ns * pinv(Wopt(:, :, n)) * H(:, :, n) * ... 
                                       Fopt(:, :, n) * Fopt(:, :, n)' * H(:, :, n)' * Wopt(:, :, n))));  
    end
end
CPU_time_IFPAD = toc;     % For accurate CPU time comment out the Spectral Efficiency calculation step.
fprintf('   "CPU time for the IFPAD method is %.4f ms." \n', CPU_time_IFPAD / Nsamples / length(NRF) * 1000);

%% OMP Hybrid Beamforming Method

SE_OMP = zeros(length(NRF), Nsamples);

tic
for n = 1:Nsamples
   for s = 1:length(NRF)
       [FRF, FBB] = OMP(Fopt(:, :, n), NRF(s), At(:,:,n));  
       [WRF, WBB] = OMP(Wopt(:, :, n), NRF(s), Ar(:,:,n)); 
       FBB = sqrt(Ns) / norm(FRF * FBB, 'fro') * FBB;

       % Spectral Efficiency
       SE_OMP(s, n) = log2(real(det(eye(Ns) + SNR / Ns * pinv(WRF * WBB) * H(:, :, n) * ...
                                    (FRF * (FBB * FBB') * FRF') * H(:, :, n)' * WRF * WBB)));           
    end
end
CPU_time_OMP = toc;
fprintf('   "CPU time for the OMP method is %.4f ms." \n', CPU_time_OMP / Nsamples / length(NRF) * 1000);

%% AO-ICD Hybrid Beamforming Method

SE_AO = zeros(length(NRF), Nsamples);

tic
for n = 1:Nsamples
    for s = 1:length(NRF)
        [FRF, FBB, WRF, WBB] = AO_ICD(H(:, :, n), Ns, NRF(s), SNR);

        % Spectral Efficiency
        SE_AO(s, n) = log2(real(det(eye(Ns) + SNR / Ns * pinv(WRF * WBB) * H(:, :, n) * ...
                                   (FRF * (FBB * FBB') * FRF') * H(:, :, n)' * WRF * WBB)));
    end
end
CPU_time_AO = toc;
fprintf('   "CPU time for the AO-ICD method is %.4f ms." \n', CPU_time_AO / Nsamples / length(NRF) * 1000);

%% MO-AltMin Hybrid Beamforming Method

SE_MO = zeros(length(NRF), Nsamples);

tic
for n = 1:Nsamples
   for s = 1:length(NRF)
       [FRF, FBB] = MO_AltMin(Fopt(:, :, n), NRF(s));  
       [WRF, WBB] = MO_AltMin(Wopt(:, :, n), NRF(s)); 
       FBB = sqrt(Ns) / norm(FRF * FBB, 'fro') * FBB;

       % Spectral Efficiency
       SE_MO(s, n) = log2(real(det(eye(Ns) + SNR / Ns * pinv(WRF * WBB) * H(:, :, n) * ...
                                    (FRF * (FBB * FBB') * FRF') * H(:, :, n)' * WRF * WBB)));           
    end
end
CPU_time_MO = toc;
fprintf('   "CPU time for the MO-AltMin method is %.4f ms." \n', CPU_time_MO / Nsamples / length(NRF) * 1000);

%% Performance Evaluation 

figure
grid on
hold on
plot(NRF, mean(SE_opt, 2),   'b-o',  'LineWidth', 2.5)
plot(NRF, mean(SE_IFPAD, 2), 'm-p',  'LineWidth', 2.5)
plot(NRF, mean(SE_MO, 2),    'k:+',  'LineWidth', 2.5)
plot(NRF, mean(SE_AO, 2),    'c-.*', 'LineWidth', 2.5)
plot(NRF, mean(SE_OMP, 2),   'g-s',  'LineWidth', 2.5)
legend('Optimal Precoder', 'IFPAD', 'MO-AltMin', 'AO-ICD', 'OMP', 'Location', 'southeast')
xlabel('NRF')
ylabel('Spectral Efficiency (bits/s/Hz)')
axis([5 10 32 44]) 
