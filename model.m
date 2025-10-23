function [feaIdx, F, W, B, E, A, S, obj] = model(X, param)
% X:dim*n
% min norm2(W'X-E)^2 + alpha*norm2(F-BE')^2 + eta*norm21(W) +
% gamma*tr(F*LS*F) + beta*norm2(S-A)^2
% s.t. E'E=I,E>0

NITER = 30;
pd = 1e4;
[dim, num] = size(X);
%% Initialize
E1 = litekmeans(X',c);
E = full(ind2vec(E1'))';Z = max(E, 0);
%B = eye(d, c);M = max(B,0);
B = orth(rand(d,c));
D = eye(dim, dim);
W1 = eye(dim, d);
F = X'*W1;
W = eye(dim, c);
%% compute A
[A, ~] = selftuning(X', k);

obj = zeros(NITER,1);
for iter = 1:NITER
    %% Update B
    [Ub, ~, Vb] = svd(E'*F);
    Imc=eye(size(Vb,1),size(Ub,1));
    B=Vb*Imc*Ub';
    %% Update W
    W = inv(X*X'+pn*D)*X*E;
    %% Update D
    dd = sqrt(sum(W.*W, 2)+eps);
    dd1 = (1/2)./dd;
    D = diag(dd1);
    %% Update S
    distf = L2_distance_1(F',F');
    S = zeros(num,num);
    for i = 1:num
        idxs0 = 1:num;
        dai = A(i, idxs0);
        dfi = distf(i, idxs0);
        ad = dai-(pr/(4*pb))*dfi;
        S(i,idxs0) = SimplexProjection(ad);
    end
    S = (S+S')/2;
    D1 = diag(sum(S));
    Ls = D1-S;
    %% Update E
    [Ue, ~, Ve] = svd(W'*X+pa*B'*F'+pd*Z');
    Imc1 = eye(size(Ve, 1), size(Ue, 1));
    E = Ve*Imc1*Ue';
    %% Update Z
    Z = (E+abs(E))/2;
    %% Update F
    II = eye(num,num);
    F = inv((1+pa)*II+pr*Ls)*(pa*E*B');

    obj(iter) = norm(X'*W-E, 'fro')^2 + pa*norm(F'-B*E', 'fro')^2 + pn*sum(sqrt(sum(W.*W, 2)+eps)) + ...
        pd*norm(Z-E, 'fro')^2 + pr*trace(F'*Ls*F) + pb*norm(S-A, 'fro')^2;
    % disp(['iter = ', num2str(iter), ', objValue = ', num2str(obj(iter))])
    
    if(iter>1 && abs((obj(iter)-obj(iter-1))/obj(iter-1))<1e-6) 
        % disp(['Ours is converge in ',num2str(iter),' iterations']);
        break;
    end
end
if iter==NITER
    % disp('Ours is not converge');
end
[~, feaIdx] = sort(sum(W.*W,2),'descend');
end

