function [mse, mse_bootstrap_90p, mse_q_matrix, mse_q_matrix_avetargets, mse_bootstrap, w] = mse_stats_given_se(se, w, n_replicates)

% se is N x N_targets x N_methods
[n, m, n_methods] = size(se);

%% set up bootstrap
if nargin < 2 || isempty(w)
    w = []; % bootstrap weights
else
    % note: first column should have equal weights (can be used to compute the actual
    % statistic), the rest do the bootstrap replicates
    assert(length(unique(w(:, 1))) == 1);
    n_replicates = size(w, 2) - 1;
end

if isempty(w) && nargin < 3
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

%%
mse_bootstrap = bsxfun(@times, se, reshape(w, [n 1 1 (n_replicates+1)]));

% average over the samples:
mse_bootstrap = squeeze(sum(mse_bootstrap, 1)); % sum instead of mean because the values have been weighted (and weights sum to 1)
% gives N_targets x N_methods x N_replicates
mse_bootstrap = permute(mse_bootstrap, [2 1 3]);
% changed to N_methods x N_targets x N_replicates

mse = mse_bootstrap(:, :, 1);
mse_bootstrap(:, :, 1) = []; % remove the non-bootstrap sample
mse_bootstrap_90p = quantile(mse_bootstrap, [0.05, 0.95], 3);

% probabilities that method along the row is better than the method along
% the column:
mse_q_matrix = zeros(m, n_methods, n_methods);
for i = 1:n_methods
    for j = 1:n_methods
        if i == j
            mse_q_matrix(:, i, j) = NaN;
            continue;
        end
        mse_q_matrix(:, i, j) = mean(mse_bootstrap(i, :, :) < mse_bootstrap(j, :, :), 3);
    end
end

mse_q_matrix_avetargets = zeros(n_methods, n_methods);
for i = 1:n_methods
    for j = 1:n_methods
        if i == j
            mse_q_matrix_avetargets(i, j) = NaN;
            continue;
        end
        mse_q_matrix_avetargets(i, j) = mean(mean(mse_bootstrap(i, :, :), 2) < mean(mse_bootstrap(j, :, :), 2));
    end
end

end