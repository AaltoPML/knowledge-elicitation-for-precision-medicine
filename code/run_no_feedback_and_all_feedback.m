function [yhat, method_names] = run_no_feedback_and_all_feedback(x_tr, y_tr, x_te, relevance_fb, directional_fb, model_hyperparams, ep_options)
% This code runs the following prediction models:
% 1) baseline (training data mean)
% 2) sparse linear regression without feedback
% 3) sparse linear regression with all feedback
% 4) sparse linear regression with all feedback except directional
method_names = {'Training data mean', ...
                'Sparse lin.reg. w/o fb', ...
                'Sparse lin.reg. w/ rel.+dir.fb', ...
                'Sparse lin.reg. w/ rel.fb'};

%% 
n_te = size(x_te, 1);
[n, m] = size(y_tr);
yhat = zeros(n_te, m, 4);

% tr data mean
yhat(:, :, 1) = ones(n_te, 1) * mean(y_tr);

% without feedback
for target_i = 1:m
    [fa, si, converged, subfuncs] = linreg_sns_ep(y_tr(:, target_i), x_tr, model_hyperparams, ep_options);
    if converged == 0
        fprintf('Run %d without feedback did not converge.\n', target_i);
    end
    yhat(:, target_i, 2) = x_te * fa.w.Mean;
end

% with all feedbacks (with and without directional)
for target_i = 1:m
    % 1st col: value; 2nd col: indices
    inds_rel = find(~isnan(relevance_fb(:, target_i)));
    gamma_feedbacks = [relevance_fb(inds_rel, target_i) inds_rel];
    inds_sign = find(~isnan(directional_fb(:, target_i)));
    wsign_feedbacks = [directional_fb(inds_sign, target_i) inds_sign];
    
    % with directional fb
    [fa, si, converged, subfuncs] = linreg_sns_ep(y_tr(:, target_i), x_tr, model_hyperparams, ep_options, [], gamma_feedbacks, wsign_feedbacks);
    if converged == 0
        fprintf('Run %d with all feedback did not converge.\n', target_i);
    end
    yhat(:, target_i, 3) = x_te * fa.w.Mean;
    
    % without directional fb
    [fa, si, converged, subfuncs] = linreg_sns_ep(y_tr(:, target_i), x_tr, model_hyperparams, ep_options, [], gamma_feedbacks, []);
    if converged == 0
        fprintf('Run %d with all feedback (no directional) did not converge.\n', target_i);
    end
    yhat(:, target_i, 4) = x_te * fa.w.Mean;
end

end