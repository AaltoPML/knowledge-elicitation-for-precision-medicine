function [model_hyperparams, ep_options] = get_model_hyperparameters_and_ep_options()

model_hyperparams.eta2 = 0.1^2; % used only if w feedback is given

% p_u/kappa has beta prior (if enabled)
model_hyperparams.kappa_prior = 1;
model_hyperparams.kappa_a = 19;
model_hyperparams.kappa_b = 1;
model_hyperparams.p_u = model_hyperparams.kappa_a / (model_hyperparams.kappa_a + model_hyperparams.kappa_b);


% rho has beta prior (if enabled)
model_hyperparams.rho_prior = 1;
model_hyperparams.rho_a = 10;
model_hyperparams.rho_b = 90;
model_hyperparams.rho = model_hyperparams.rho_a / (model_hyperparams.rho_a + model_hyperparams.rho_b);

% 1/sigma2 has gamma prior (if enabled)
model_hyperparams.sigma2 = 1;
model_hyperparams.sigma2_prior = 1;
model_hyperparams.sigma2_a = 4;
model_hyperparams.sigma2_b = 4;

% log(tau2) has normal prior (if enabled)
model_hyperparams.tau2 = 1;
model_hyperparams.tau2_prior = 1;
model_hyperparams.tau2_tau = 2; % precision
model_hyperparams.tau2_mu = -6.5; % note: this is precision-adj. mean
model_hyperparams.tau2_shared = 0;

ep_options = struct('damp', 0.05, ...
                    'damp_decay', 1, ...
                    'robust_updates', 2, ...
                    'verbosity', 0, ...
                    'max_iter', 1000,...
                    'gamma_threshold', 1e-4,...
                    'w_threshold', 1e-5,...
                    'min_site_prec', 1e-6,...
                    'max_site_prec', Inf, ...
                    'w_mean_update_threshold', 1e-6,...
                    'w_prec_update_threshold', 1e6,...
                    'hermite_n', 11, ...
                    'degenerate_representation', 1);

if ~isfield(ep_options, 'hermite_x') % assume that if locations are given, weights will also be given
    % Gauss-Hermite quadrature: using the weights and eval.locations from
    % EKF/UKF toolbox (http://becs.aalto.fi/en/research/bayes/ekfukf/)
    h_n = ep_options.hermite_n;
    h_p = hermitepolynomial(h_n);
    ep_options.hermite_x = roots(h_p);
    
    h_Wc = pow2(h_n-1) * factorial(h_n) * sqrt(pi) / h_n^2;
    h_p2 = hermitepolynomial(h_n - 1);
    ep_options.hermite_W  = zeros(h_n, 1);
    for i = 1:h_n
        ep_options.hermite_W(i) = h_Wc * polyval(h_p2, ep_options.hermite_x(i)).^-2;
    end
    ep_options.hermite_W = ep_options.hermite_W / sqrt(pi);
end

end