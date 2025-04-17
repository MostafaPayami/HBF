%% Channel Generation of mmWave Massive MIMO-OFDM systems

% clear
% clc

% rng(700);
rng('shuffle');

Nt  = 256;      % Number of transmit antennas
Nr  = 64;       % Number of receive antennas
K   = 128;      % Number of OFDM subcarriers
Ns  = 5;        % Number of data streams
NRF = 7;        % Number of RF chains
Nc  = 6;        % Number of clusters
Nray = 12;      % Number of rays in each cluster
angle_sigma = 15 / 180 * pi;        % Tx/Rx azimuth and elevation anglular spread
gamma = sqrt((Nt*Nr) / (Nc*Nray));  % Normalization factor
sigma = 1;                          % Normalization condition of mmWave channel H

Nsample = 100;   % Number of channel realizations
count   = 0;
Fopt = zeros(Nt, Ns, K, Nsample);
Wopt = zeros(Nr, Ns, K, Nsample);
H  = zeros(Nr, Nt, K, Nsample);
At = zeros(Nt, Nc*Nray, Nsample);
Ar = zeros(Nr, Nc*Nray, Nsample);

%%
tic
for n = 1:Nsample
    % H = zeros(Nr, Nt, K);
    Ht = zeros(Nr, Nt, Nc);
    for c = 1:Nc
        AoD_m = unifrnd(0, 2*pi, [1, 2]);
        AoA_m = unifrnd(0, 2*pi, [1, 2]);
        
        AoD(1, :) = laprnd(1, Nray, AoD_m(1), angle_sigma);
        AoD(2, :) = laprnd(1, Nray, AoD_m(2), angle_sigma);
        AoA(1, :) = laprnd(1, Nray, AoA_m(1), angle_sigma);
        AoA(2, :) = laprnd(1, Nray, AoA_m(2), angle_sigma);        
        for j = 1:Nray
            temp = (c-1) * Nray + j;
            At(:, temp, n) = array_response(AoD(1, j), AoD(2, j), Nt);
            Ar(:, temp, n) = array_response(AoA(1, j), AoA(2, j), Nr);
            alpha = normrnd(0, sqrt(sigma/2)) + 1i * normrnd(0, sqrt(sigma/2));
            Ht(:, :, c) = Ht(:, :, c) + alpha * Ar(:, temp, n) * At(:, temp, n)';
        end
    end
    
    for k = 1:K
        for c = 1:Nc
            H(:, :, k, n) = H(:, :, k, n) + Ht(:, :, c) * exp(-1i * 2 * pi * (k-1) / K * (c-1));
        end
        H(:, :, k, n) = H(:, :, k, n) * gamma;             
        if (rank(H(:, :, k, n) ) >= Ns)
            [U, ~, V] = svd(H(:, :, k, n), "econ");
            Fopt(:, :, k, n) = V(:, 1:Ns);     % Optimal Precoder
            Wopt(:, :, k, n) = U(:, 1:Ns);     % Optimal Combiner
            count = count + 1;            
        end
    end
end
CPU_time_CR = toc;
fprintf('  "CPU time for generating OFDM Channel Realizations is %f seconds." \n', CPU_time_CR);
