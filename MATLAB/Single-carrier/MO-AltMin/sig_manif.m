function [y, cost] = sig_manif(Fopt, FRF, FBB)
[Nt, NRF] = size(FRF);

manifold = complexcirclefactory(Nt*NRF);
problem.M = manifold;

%% Efficient Implementation
problem.cost  = @(x) norm(Fopt - reshape(x, Nt, NRF) * FBB, 'fro')^2;
problem.egrad = @(x) reshape(-2 * (Fopt - reshape(x, Nt, NRF) * FBB) * FBB', [], 1);

%% Original Implementation
% % problem.cost  = @(x) norm(Fopt - reshape(x, Nt, NRF) * FBB, 'fro')^2;
% % problem.egrad = @(x) -2 * kron(conj(FBB), eye(Nt)) * (Fopt(:) - kron(FBB.', eye(Nt)) * x);
% 
% f = Fopt(:);
% A = kron(FBB.', eye(Nt));
% 
% problem.cost  = @(x) (f-A*x)'*(f-A*x);
% problem.egrad = @(x) -2*A'*(f-A*x);
%%

% checkgradient(problem);
warning('off', 'manopt:getHessian:approx');

%% (Added 2025/04/14)
options = struct();
options.verbosity = 0;  % Minimal output information
%% 

[x, cost, info, options] = conjugategradient(problem, FRF(:), options);
% [x,cost,info,options] = trustregions(problem, FRF(:));
% info.iter
y = reshape(x,Nt,NRF);

end