function [cinds, cinds_bootstrap_90p, cinds_q_matrix, cinds_q_matrix_avetargets, cinds_bootstrap, w] = cindex_stats(y, yhat, w, n_replicates)

% y is N x N_targets
% yhat is N x N_targets x N_methods
[n, m] = size(y);
n_methods = size(yhat, 3);

%% set up bootstrap
if nargin < 3 || isempty(w)
    w = []; % bootstrap weights
else
    % note: first column should have equal weights (can be used to compute the actual
    % statistic), the rest do the bootstrap replicates
    assert(length(unique(w(:, 1))) == 1);
    n_replicates = size(w, 2) - 1;
end

if isempty(w) && nargin < 4
    n_replicates = 10000; % bootstrap
end

% note 1: we will use the same weights for each method (allows to do
% comparisons between methods based on the boostrap replicates);
% we also use same weights for each criterion and each drug (not so important...)
% note 2: first column has equal weights (can be used to compute the actual
% statistic), the rest do the bootstrap replicates
if isempty(w)
    w = [ones(n, 1) / n dirrand(n, n_replicates)];
end
assert(size(w, 1) == n);

%% c-index (simple c-index; no probabilistic weighting or such)
cinds = zeros(n_methods, m);
cinds_bootstrap = zeros(n_methods, m, n_replicates);

for m_i = 1:n_methods
    for d_i = 1:m
        numer = zeros(1, size(w, 2));
        denom = zeros(1, size(w, 2));
        
        for i = 1:(n-1)
            for j = (i+1):n
                w_ = w(i, :) .* w(j, :);
                
                % if original has tie, skip; if prediction is tied, add half
                if y(i, d_i) == y(j, d_i) || yhat(i, d_i, m_i) == yhat(j, d_i, m_i)
                    if y(i, d_i) == y(j, d_i)
                        continue
                    end
                    numer = numer + w_ * 0.5;
                    denom = denom + w_;
                    continue
                end
                
                i_gt_j_1 = y(i, d_i) > y(j, d_i);
                i_gt_j_2 = yhat(i, d_i, m_i) > yhat(j, d_i, m_i);
                
                numer = numer + w_ * (i_gt_j_1 == i_gt_j_2);
                denom = denom + w_;
            end
        end
        
        c_ = numer ./ denom;
        
        cinds(m_i, d_i) = c_(1);
        cinds_bootstrap(m_i, d_i, :) = c_(2:end);
    end
end
cinds_bootstrap_90p = quantile(cinds_bootstrap, [0.05, 0.95], 3);

% probabilities that method along the row is better than the method along
% the column:
% separately for each target
cinds_q_matrix = zeros(m, n_methods, n_methods);
for i = 1:n_methods
    for j = 1:n_methods
        if i == j
            cinds_q_matrix(:, i, j) = NaN;
            continue;
        end
        cinds_q_matrix(:, i, j) = mean(cinds_bootstrap(i, :, :) > cinds_bootstrap(j, :, :) , 3);
    end
end

% average c-index over targets
cinds_q_matrix_avetargets = zeros(n_methods, n_methods);
for i = 1:n_methods
    for j = 1:n_methods
        if i == j
            cinds_q_matrix_avetargets(i, j) = NaN;
            continue;
        end
        cinds_q_matrix_avetargets(i, j) = mean(mean(cinds_bootstrap(i, :, :), 2) > mean(cinds_bootstrap(j, :, :), 2));
    end
end

end
