function [FRF, FBB] = MO_AltMin(Fopt, NRF) % , phase_Fopt, Fopt_2Ns

[Nt, Ns] = size(Fopt);
y = [];

% FRF = exp(1i * angle(Fopt_start));
FRF = exp(1i * unifrnd(0, 2*pi, Nt, NRF));
% Wn = 1/sqrt(NRF) * dftmtx(NRF);
% Wn = Wn(1:Ns, :);
% Fopt_NRF = Fopt * Wn;
% FRF = exp(1i * angle(Fopt_NRF));
% FRF = Fopt_NRF ./ abs(Fopt_NRF);
% T0 = unifrnd(0, 2*pi, Nt, NRF);
% T0(:, 1:Ns) = angle(Fopt);
% FRF = exp(1i * T0);
% FRF = exp(1i * pi * phase_Fopt);
% FRF = FRF_omp;
% itr = 0;

while (isempty(y) || abs(y(1)-y(2))>1e-3)

    % itr = itr + 1;
    FBB = pinv(FRF) * Fopt;
    y(1) = norm(Fopt - FRF * FBB,'fro')^2;
    [FRF, y(2)] = sig_manif(Fopt, FRF, FBB);
    
end

% for kk=1:2
% 
%     FBB = pinv(FRF) * Fopt;
%     y(1) = norm(Fopt - FRF * FBB,'fro')^2;
%     [FRF, y(2)] = sig_manif(Fopt, FRF, FBB);
% 
% end

end