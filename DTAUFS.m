function [feaIdx, F, W, B, E, S, obj] = DTAUFS(X, param, S)
% X: dim * n
% min alpha*norm(X'*W-E, 'fro')^2 + norm(F'-B*E', 'fro')^2
%  + gamma*trace(F'*Ls*F) + eta*norm(W, 2,1) + delta*norm(Z-E, 'fro')^2
% s.t. E'*E=I, E>=0, B'*B=I

% % Compute fixed graph S
% [A, ~] = selftuning(X', k);
% S = (A + A') / 2;
% D1 = diag(sum(S));
% Ls = D1 - S;
% eigenvalues = eig(Ls);
% sorted_eigenvalues = sort(real(eigenvalues), 'ascend');
% lambda_c_plus_1 = sorted_eigenvalues(c + 1);
% empirical_gamma = 2 / (lambda_c_plus_1 + eps);
% r = empirical_gamma;
% r = min(r, 10e3);
% r = max(r, 10e-3);

    c = param.c;
    d = param.d;
    pa = param.pa;
    pn = param.pn;
    pr = param.pr;
    pd = 1e4;
    NITER = 30;
    
    [dim, num] = size(X);
    
    %% Initialize
    E1 = litekmeans(X', c);
    E = full(ind2vec(E1'))';
    Z = max(E, 0);
    B = orth(rand(d, c));
    D = eye(dim, dim);
    W1 = eye(dim, d);
    F = X' * W1;
    W = eye(dim, c);
    
    %% Compute S
    D1 = diag(sum(S));
    Ls = D1 - S;
    
    II = eye(num, num); 
    obj = zeros(NITER, 1);
    
    for iter = 1:NITER
        %% Update B
        [Ub, ~, Vb] = svd(E' * F, 'econ');
        B = Vb * Ub';
        
        %% Update W
        epsilon = 1e-8;
        I = eye(size(X, 1));
        W = (X * X' + (pn / pa) * D + epsilon * I) \ (X * E);
        
        %% Update D
        dd = sqrt(sum(W .* W, 2) + eps);
        dd1 = (1 / 2) ./ dd;
        D = diag(dd1);
        
        %% Update E
        [Ue, ~, Ve] = svd(pa * W' * X + B' * F' + pd * Z', 'econ');
        E = Ve * Ue';
        
        %% Update Z
        Z = (E + abs(E)) / 2;
        
        %% Update F
        F = (II + pr * Ls) \ (E * B');
        
        %% Objective Function
        obj(iter) = pa * norm(X' * W - E, 'fro')^2 + ...
                    norm(F' - B * E', 'fro')^2 + ...
                    pn * sum(dd) + ...
                    pd * norm(Z - E, 'fro')^2 + ...
                    pr * trace(F' * Ls * F);
        
        %% Convergence Check
        if (iter > 1 && abs((obj(iter) - obj(iter - 1)) / max(1, obj(iter - 1))) < 1e-6)
            obj = obj(1:iter); 
            break;
        end
    end
    
    [~, feaIdx] = sort(sum(W .* W, 2), 'descend');
end
