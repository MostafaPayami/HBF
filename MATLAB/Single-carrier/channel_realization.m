%% Hybrid Precoding for mmWave Massive MIMO Wireless Systems 

% % This code realizes mmWave channels

% clear 
clc

% addpath(genpath(pwd));

rng(1)
% rng('shuffle') 

tic

Nt  = 256;             % Number of transmit antennas
Nr  = 64;              % Number of receive antennas
Ns  = 4;               % Number of data streams
NRF = 4;    % Number of RF chains 

Nc = 6;                               % Number of clusters
Nray = 12;                            % Number of rays in each cluster
angle_sigma = 15 / 180 * pi;          % Standard deviation of azimuth and elevation angles (Tx and Rx)
gamma = sqrt((Nt*Nr) / (Nc*Nray));    % Normalization factor
sigma = 1;                            % According to the normalization condition of the H

Nsample = 1000;
count = 0;

H  = zeros(Nr, Nt, Nsample);
S  = zeros(Nr, Nr, Nsample);
At = zeros(Nt, Nc*Nray, Nsample);
Ar = zeros(Nr, Nc*Nray, Nsample);
alpha = zeros(Nc*Nray, Nsample);

Fopt = zeros(Nt, Ns, Nsample);
Wopt = zeros(Nr, Ns, Nsample);

AoD_m = zeros(2, Nc, Nsample);
AoA_m = zeros(2, Nc, Nsample);

AoD = zeros(2, Nc*Nray, Nsample);
AoA = zeros(2, Nc*Nray, Nsample);

for n = 1:Nsample
    for c = 1:Nc
        AoD_m(:, c, n) = unifrnd(0, 2*pi, 1, 2);      % Mean of AoD for phi and theta
        AoA_m(:, c, n) = unifrnd(0, 2*pi, 1, 2);      % Mean of AoA for phi and theta
        
        AoD(1, (c-1)*Nray+1:Nray*c, n) = laprnd(1, Nray, AoD_m(1, c, n), angle_sigma);
        AoD(2, (c-1)*Nray+1:Nray*c, n) = laprnd(1, Nray, AoD_m(2, c, n), angle_sigma);

        AoA(1, (c-1)*Nray+1:Nray*c, n) = laprnd(1, Nray, AoA_m(1, c, n), angle_sigma);
        AoA(2, (c-1)*Nray+1:Nray*c, n) = laprnd(1, Nray, AoA_m(2, c, n), angle_sigma);
    end
    
    H(:, :, n) = zeros(Nr, Nt);
    for j = 1:Nc*Nray
        At(:, j, n) = array_response(AoD(1,j,n), AoD(2,j,n), Nt);      % UPA transmit array response
        Ar(:, j, n) = array_response(AoA(1,j,n), AoA(2,j,n), Nr);      % UPA receive array response
        alpha(j, n) = normrnd(0, sqrt(sigma/2)) + 1i * normrnd(0, sqrt(sigma/2));
        H(:, :, n)  = H(:, :, n) + alpha(j, n) * Ar(:, j, n) * At(:, j, n)';
    end
    H(:, :, n) = gamma * H(:, :, n);
    
    if(rank(H(:, :, n)) >= Ns)
        count = count + 1;
        [U, S(:, :, n), V] = svd(H(:, :, n), "econ");
        % [U(:, :, reali), S, V(:, :, reali)] = svd(H(:, :, reali));

        Fopt(:, :, n) = V(1:Nt, 1:Ns);
        Wopt(:, :, n) = U(1:Nr, 1:Ns);

    end
end

toc