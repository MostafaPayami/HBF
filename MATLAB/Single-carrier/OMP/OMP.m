function [FRF, FBB, AoD_indx] = OMP( Fopt, NRF, At )

FRF = [];
b0 = zeros(NRF, 1);
Fres = Fopt;
for k = 1:NRF
    PU = At' * Fres;
%     [aa,bb] = max(diag(PU * PU'));
    [a0, b0(k)] = max(sum(abs(PU).^2, 2));
    FRF = [FRF , At(:, b0(k))];
    FBB = pinv(FRF) * Fopt;   
    Fres = (Fopt - FRF * FBB) / norm(Fopt - FRF * FBB,'fro');
end
AoD_indx = b0;
end

