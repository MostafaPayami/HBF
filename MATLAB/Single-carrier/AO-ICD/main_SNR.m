%% AO-ICD Algorithm (2016)

% clear
% clc

% load('Hybrid_Beamforming_mmWave_Massive_MIMO.mat');
% load('Ns=8, 2000.mat');

tic

% Ns  = 4;         % Number of data streams
NRF = 4;         % Number of RF chains 

SNRdB = -35:5:10;             % Signal to Noise  ratio (dB)
SNR   = 10 .^ (SNRdB / 10);
Nsample = size(H, 3);
smax = length(SNR);           % Enable the parallel

FRF_ao = zeros(Nt,  NRF, Nsample);
FBB_ao = zeros(NRF, Ns,  Nsample);

WRF_ao = zeros(Nr,  NRF, Nsample);
WBB_ao = zeros(NRF, Ns,  Nsample);

R_AO = zeros(smax, Nsample);

for n = 1:Nsample
    n
    for s = 1:smax
       [FRF_ao, FBB_ao, WRF_ao, WBB_ao] = AO_ICD(H(:, :, n), Ns, NRF, SNR(s));
       % [FRF_ao, FBB_ao, WRF_ao, WBB_ao, iterF(n), iterW(n)] = AO_ICD(H(:, :, n), Ns, NRF, SNR(s));

        R_AO(s, n) = log2(det(eye(Ns) + SNR(s)/Ns * pinv(WRF_ao * WBB_ao) * ...
                          H(:, :, n) * FRF_ao * (FBB_ao * FBB_ao') * ...
                          FRF_ao' * H(:, :, n)' * WRF_ao * WBB_ao));

    end
end

CPU_time_AO = toc;
fprintf('\n   "CPU time for AO-ICD method is %.4e seconds." \n \n', CPU_time_AO);

%% Plot

% figure
plot(SNRdB, mean(R_AO, 2), 'c-+', 'LineWidth', 2)
% grid on
% hold on
