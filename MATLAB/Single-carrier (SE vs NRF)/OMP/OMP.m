%% OMP Hybrid Beamforming Method

function [FRF, FBB] = OMP(Fopt, NRF, At)
FRF = [];
Fres = Fopt;

for k = 1:NRF
    PU = At' * Fres;
    % [aa,bb] = max(diag(PU * PU'));
    [~, bb] = max(sum(abs(PU).^2, 2));
    FRF = [FRF , At(:, bb)];
    FBB = pinv(FRF) * Fopt;   
    Fres = (Fopt - FRF * FBB) / norm(Fopt - FRF * FBB,'fro');
end
end

