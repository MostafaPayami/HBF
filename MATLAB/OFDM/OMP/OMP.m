function [FRF, FBB] = OMP(Fopt, NRF, At)

[Nt, Ns, K] = size(Fopt);
FRF = [];
FBB = zeros(NRF, Ns, K);
Fres = Fopt;

for n = 1:NRF
    %% Original implementation 
    % temp = 0;
    % for k = 1:K
    %     PU(:,:,k) = At' * Fres(:,:,k);
    %     temp = temp + sum( abs(PU(:,:,k)).^2, 2 );        
    % end
    % [aa,bb] = max(temp);
    % FRF = [FRF , At(:,bb)];
    % for k = 1:K
    %     FBB(1:i,:,k) = pinv(FRF) * Fopt(:,:,k);
    %     Fres(:,:,k) = (Fopt(:,:,k) - FRF * FBB(1:i,:,k)) / norm(Fopt(:,:,k) - FRF * FBB(1:i,:,k),'fro');
    % end

    %% Efficient implementation 
    PU = pagemtimes(At, 'ctranspose', Fres, 'none'); 
    CP = sum(abs(PU).^2, [2, 3]);   % Correlation Proxy
    [~, index] = max(CP);
    FRF = [FRF, At(:, index)];
    FBB(1:n, :, :) = pagemtimes(pinv(FRF), Fopt);
    dF = Fopt - pagemtimes(FRF, FBB(1:n, :, :)); 
    Fres = dF ./ (pagenorm(dF, 'fro') + eps);    
end
end
