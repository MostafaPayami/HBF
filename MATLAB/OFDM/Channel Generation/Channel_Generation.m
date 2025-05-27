%% Random Channel Realizations of mmWave Massive MIMO-OFDM Systems 

function [Fopt, Wopt, H, At, Ar] = Channel_Generation(Nt, Nr, K, Ns, Nsamples) 
%% Channel Parameters  

Nc = 6;                             % Number of clusters
Nray = 12;                          % Number of rays in each cluster
angle_sigma = 15 / 180 * pi;        % Azimuth and Elevation angle spreads (Tx and Rx)
gamma = sqrt((Nt*Nr) / (Nc*Nray));  % Normalization factor
sigma = 1;                          % Normalization condition of mmWave channel

%% Channel Generation

count = 0;
H  = zeros(Nr, Nt, K, Nsamples);
At = zeros(Nt, Nc*Nray, Nsamples);
Ar = zeros(Nr, Nc*Nray, Nsamples);
Fopt = zeros(Nt, Ns, K, Nsamples);
Wopt = zeros(Nr, Ns, K, Nsamples);

for n = 1:Nsamples
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
end