function [x, ft] = SimplexProjection(y)

%   min_x  0.5 * ||x - y||^2
%   s.t.   sum(x) = 1, x >= 0
    
    y = y';
    u = sort(y, 'descend');  
    sv = cumsum(u);
    rho = find(u + (1 - sv) ./ (1:length(u))' > 0, 1, 'last');
    lambda = (1 - sv(rho)) / rho;
    x = max(y + lambda, 0);
    ft = 0.5 * norm(x - y)^2;
end
