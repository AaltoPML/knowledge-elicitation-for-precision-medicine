%% This is a simple example demonstrating the use of the code
addpath('../code')
rng(1234)

% data generation parameters
N_tr = 50;       % training samples
N_te = 200;      % test samples
Dx = 100;        % number of covariates
p_rel = 10 / Dx; % probability of covariate being relevant
Dy = 3;          % number of outputs
s2 = 0.5;        % noise variance (residual noise ~ N(0, s2))
t2 = 0.05;       % signal variance (reg.weight ~ N(0, t2))
p_u_fn = 0.01;   % flip noise probability for user's feedback
                 % (0 = all feedback are correct)
p_u_dk = 0.25;   % probability of don't know responses (0 = know all)

% note: prior parameters and EP options for the sparse linear regression
% models are in get_model_hyperparameters_and_ep_options.m. One might need
% to update these according to your prior beliefs of the data generation.
[model_hyperparams, ep_options] = get_model_hyperparameters_and_ep_options();

% generate simulated data
x_tr = randn(N_tr, Dx);
x_te = randn(N_te, Dx);
B = zeros(Dx, Dy);
Bmask = rand(Dx, Dy) < p_rel;
B(Bmask) = sqrt(t2) * randn(sum(Bmask(:)), 1);

y_tr = x_tr * B + sqrt(s2) * randn(N_tr, Dy);
y_te = x_te * B + sqrt(s2) * randn(N_te, Dy);

% generate simulated feedback
% (don't knows should be nans)
relevance_fb = 1 * Bmask; % 0/1

% add flip noise
noisy_fb = rand(prod(size(relevance_fb)), 1) < p_u_fn;
relevance_fb(noisy_fb) = 1 - relevance_fb(noisy_fb);

% add don't knows
dont_know = rand(prod(size(relevance_fb)), 1) < p_u_dk;
relevance_fb(dont_know) = nan;

% directional feedback (simulate only for those with relevance feedback)
directional_fb = nan(size(relevance_fb));
dfb_mask = relevance_fb == 1 & B ~= 0;
directional_fb(dfb_mask) = 2 * (B(dfb_mask) > 0)- 1; % +1/-1

% add flip noise (among only those that are present)
noisy_fb = rand(sum(~isnan(directional_fb(:))), 1) < p_u_fn;
directional_fb(noisy_fb) = -1 * directional_fb(noisy_fb);

% add don't knows (among those that are otherwise known)
dont_know = rand(sum(~isnan(directional_fb(:))), 1) < p_u_dk;
directional_fb(dont_know) = nan;

%% Run non-sequential analysis. Gives predictions for
% 1) baseline (training data mean)
% 2) sparse linear regression without feedback
% 3) sparse linear regression with all feedback
% 4) sparse linear regression with all feedback except directional
[yhat, method_names] = run_no_feedback_and_all_feedback(x_tr, y_tr, x_te, relevance_fb, directional_fb, model_hyperparams, ep_options);

% Print out MSEs
mse = squeeze(mean(bsxfun(@minus, y_te, yhat).^2, 1))';
mse = [mse mean(mse, 2)];
outcome_names = cell(Dy + 1, 1); for i = 1:Dy, outcome_names{i} = sprintf('Outcome_%d', i); end
outcome_names{Dy + 1} = 'Average';
mse = array2table(mse, 'VariableNames', outcome_names, 'RowNames', method_names);
disp(mse);

%% Run sequential analysis (warning: the target-specific infogain can be slow)
n_iter = 50; % run up to 50 feedbacks

% random sequence
method = 'random';
yhat_random = run_infogain(x_tr, y_tr, x_te, relevance_fb, directional_fb, model_hyperparams, ep_options, method, n_iter);

% information gain evaluated over training set
method = 'tr_infogain';
yhat_tr_infogain = run_infogain(x_tr, y_tr, x_te, relevance_fb, directional_fb, model_hyperparams, ep_options, method, n_iter);

% information gain evaluated over test set (target-specific)
% Warning: this can be slow when run for many targets and large number of
% feedbacks
method = 'te_infogain';
yhat_te_infogain = zeros(n_iter + 1, Dy, N_te);
for i = 1:N_te
    yhat_te_infogain(:, :, i) = run_infogain(x_tr, y_tr, x_te(i, :), relevance_fb, directional_fb, model_hyperparams, ep_options, method, n_iter);
end

%% plot MSEs
mse_random = squeeze(mean(bsxfun(@minus, permute(y_te, [3 2 1]), yhat_random).^2, 3));
mse_random = [mse_random mean(mse_random, 2)];

mse_tr_infogain = squeeze(mean(bsxfun(@minus, permute(y_te, [3 2 1]), yhat_tr_infogain).^2, 3));
mse_tr_infogain = [mse_tr_infogain mean(mse_tr_infogain, 2)];

mse_te_infogain = squeeze(mean(bsxfun(@minus, permute(y_te, [3 2 1]), yhat_te_infogain).^2, 3));
mse_te_infogain = [mse_te_infogain mean(mse_te_infogain, 2)];

figure(1); clf;
hs = [];
for i = 1:(Dy + 1)
    subplot((Dy + 1) / 2, 2, i);
    plot(0:n_iter, mse_random(:, i), 'k-'); hold on;
    plot(0:n_iter, mse_tr_infogain(:, i), 'm-');
    plot(0:n_iter, mse_te_infogain(:, i), 'r-');
    title(outcome_names{i}, 'interpreter', 'none');
    xlabel('number of feedbacks');
    ylabel('MSE');
    legend('Random sequential sampling', 'Sequetial experimental design', 'Targeted sequential experimental design', 'Location', 'SouthWest');
end


