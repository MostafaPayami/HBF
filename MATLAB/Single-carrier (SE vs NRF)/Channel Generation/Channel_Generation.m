%% Random Channel Realizations of mmWave Massive MIMO Systems 

function [Fopt, Wopt, H, At, Ar] = Channel_Generation(Nt, Nr, Ns, Nsamples) 
%% Channel Parameters  

Nc = 6;                             % Number of clusters
Nray = 12;                          % Number of rays in each cluster
angle_sigma = 15 / 180 * pi;        % Azimuth and Elevation angle spreads (Tx and Rx)
gamma = sqrt((Nt*Nr) / (Nc*Nray));  % Normalization factor
sigma = 1;                          % Normalization condition of mmWave channel

%% Channel Generation

count = 0;
H  = zeros(Nr, Nt, Nsamples);
At = zeros(Nt, Nc*Nray, Nsamples);
Ar = zeros(Nr, Nc*Nray, Nsamples);
alpha = zeros(Nc*Nray, Nsamples);

Fopt = zeros(Nt, Ns, Nsamples);
Wopt = zeros(Nr, Ns, Nsamples);

AoD_m = zeros(2, Nc, Nsamples);
AoA_m = zeros(2, Nc, Nsamples);

AoD = zeros(2, Nc*Nray, Nsamples);
AoA = zeros(2, Nc*Nray, Nsamples);

for n = 1:Nsamples
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
        [U, ~, V] = svd(H(:, :, n), "econ");
        Fopt(:, :, n) = V(1:Nt, 1:Ns);
        Wopt(:, :, n) = U(1:Nr, 1:Ns);
    end
end
