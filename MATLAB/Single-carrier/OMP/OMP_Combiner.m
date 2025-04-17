function [ WRF, WBB ] = OMP_Combiner( Wmmse, NRF, Ar , Eyy)

WRF = [];
Wres = Wmmse;      % W_optimal = W_MMSE
for k = 1:NRF
    PU = Ar' * Eyy * Wres;
%     [aa,bb] = max(diag(PU * PU'));
    [aa,bb] = max(sum( abs(PU).^2, 2 ));
    WRF = [WRF , Ar(:,bb)];
    WBB = inv(WRF' * Eyy * WRF) * WRF' * Eyy * Wmmse;  
    Wres = (Wmmse - WRF * WBB) / norm(Wmmse - WRF * WBB,'fro');
end

end

