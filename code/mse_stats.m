function [mse, mse_bootstrap_90p, mse_q_matrix, mse_q_matrix_avetargets, mse_bootstrap, w] = mse_stats(y, yhat, w, n_replicates)

% y is N x N_targets
% yhat is N x N_targets x N_methods
[n, m, n_methods] = size(yhat);

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

%%
se = bsxfun(@minus, yhat, y).^2;
[mse, mse_bootstrap_90p, mse_q_matrix, mse_q_matrix_avetargets, mse_bootstrap, w] = mse_stats_given_se(se, w, n_replicates);

end