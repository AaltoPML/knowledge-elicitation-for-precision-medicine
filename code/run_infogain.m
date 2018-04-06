function [yhat, queried_features, queried_targets] = run_infogain(x_tr, y_tr, x_te, relevance_fb, directional_fb, model_hyperparams, ep_options, method, n_iter, verbosity)
% This code runs a selected sequential experimental design algorithm for a
% given set of feedbacks and number of iterations of querying the user.

if nargin < 10
    verbosity = 0;
end

switch method
    case 'random'
        random_sel = true;
        tr_infogain = false; % doesn't matter which if random selection
    case 'tr_infogain'
        random_sel = false;
        tr_infogain = true;
    case 'te_infogain'
        random_sel = false;
        tr_infogain = false;
    otherwise
        error('unknown method num');
end

%%
if tr_infogain
    x_util = x_tr;
else
    x_util = x_te;
end

n_te = size(x_te, 1);
[n, m] = size(y_tr);
M = size(x_tr, 2);
yhat = zeros(n_iter + 1, m, n_te);

Utility_matrix = zeros(M, m); % #features x #targets
gamma_feedbacks = cell(m, 1);
wsign_feedbacks = cell(m, 1);
posterior_approx = cell(m, 1);
queried_features = cell(m, 1);
queried_targets = zeros(n_iter, 1);

% compute utilities and predictions before any feedback
for j = 1:m
    posterior_approx{j} = get_posterior(y_tr(:, j), x_tr, model_hyperparams, ep_options, gamma_feedbacks{j}, wsign_feedbacks{j});
    if random_sel
        Utility_matrix(:, j) = rand(M, 1);
    else
        Utility_matrix(:, j) = compute_utilities(posterior_approx{j}, x_util, model_hyperparams, ep_options);
    end
    
    yhat(1, j, :) = x_te * posterior_approx{j}.mean;
end

st = tic;
for iter = 1:n_iter
    % choose next query
    [~, I] = max(Utility_matrix(:));
    [feature_index, target_index] = ind2sub(size(Utility_matrix), I);
    queried_features{target_index} = [queried_features{target_index}; feature_index];
    queried_targets(iter) = target_index;
    
    % observe feedback
    gamma_feedback = relevance_fb(feature_index, target_index);
    wsign_feedback = directional_fb(feature_index, target_index);
    
    if isnan(gamma_feedback) && isnan(wsign_feedback)
        dont_know_feedback = true;
    else
        dont_know_feedback = false;
        
        if ~isnan(gamma_feedback)
            gamma_feedbacks{target_index} = [gamma_feedbacks{target_index}; gamma_feedback feature_index];
        end
        if ~isnan(wsign_feedback)
            wsign_feedbacks{target_index} = [wsign_feedbacks{target_index}; wsign_feedback feature_index];
        end
    end
    
    % update model
    if ~dont_know_feedback
        posterior_approx{target_index} = get_posterior(y_tr(:, target_index), x_tr, model_hyperparams, ep_options, gamma_feedbacks{target_index}, wsign_feedbacks{target_index});
        
        if posterior_approx{target_index}.converged == 0
            fprintf('Run %d at iter %d did not converge.\n', target_index, iter);
        end
    end
    
    % update predictions
    % note: in principle, we only need to update target_index prediction
    for j = 1:m
        yhat(iter + 1, j, :) = x_te * posterior_approx{j}.mean;
    end
    
    % update utilities
    if ~random_sel
        Utility_matrix(:, target_index) = compute_utilities(posterior_approx{target_index}, x_util, model_hyperparams, ep_options);
    end
    % remove those that already have already been queried
    Utility_matrix(queried_features{target_index}, target_index) = -inf;
    
    if verbosity > 0 && mod(iter, verbosity) == 0
        fprintf('Run %d of %d completed, time %d sec.\n', iter, n_iter, round(toc(st)));
    end
end

end

function posterior = get_posterior(y, x, model_hyperparams, ep_options, gamma_feedbacks, wsign_feedbacks)

[fa, si, converged, subfuncs] = linreg_sns_ep(y, x, model_hyperparams, ep_options, [], gamma_feedbacks, wsign_feedbacks);

posterior.si = si;
posterior.fa = fa;
if ep_options.degenerate_representation
    tmp_diag = diag(1./ fa.w.Tau_diag);
    tmp_lt = fa.w.degenerate_inner_chol \ (fa.w.Tau_x_half * tmp_diag);
    posterior.sigma = tmp_diag - tmp_lt' * tmp_lt;
else
    posterior.sigma = fa.w.Tau_chol' \ (fa.w.Tau_chol \ eye(size(fa.w.Tau_chol)));
end
posterior.mean = fa.w.Mean;
posterior.p = fa.gamma.p;
posterior.ep_subfunctions = subfuncs;
posterior.converged = converged;

end