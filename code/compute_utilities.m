function utility = compute_utilities(posterior, x, model_hyperparams, ep_options)

pr = model_hyperparams;
if ~isfield(pr, 'tau2_prior')
    pr.tau2_prior = 0;
end

op = ep_options;
op.damp = 1; % don't damp updates?

if isfield(pr, 'kappa_prior') && pr.kappa_prior
    p_u = posterior.fa.kappa.a / (posterior.fa.kappa.a + posterior.fa.kappa.b);
else
    p_u = pr.p_u;
end

post_pred_f0 = p_u + posterior.p - 2 * p_u * posterior.p;
%posterior predictive of feedback on sign (+)
cdf_w_less0 = normcdf(0, posterior.mean, sqrt(diag(posterior.sigma)));
post_pred_fP = p_u * (1 - cdf_w_less0) + (1 - p_u) * cdf_w_less0;

KL_P0 = compute_post_pred_kl(1, 0, posterior, pr, op, x');
KL_P1 = compute_post_pred_kl(1, 1, posterior, pr, op, x');
KL_N0 = compute_post_pred_kl(-1, 0, posterior, pr, op, x');
KL_N1 = compute_post_pred_kl(-1, 1, posterior, pr, op, x');

utility = (1-post_pred_f0) .* post_pred_fP .* KL_P1 + post_pred_f0 .* post_pred_fP .* KL_P0 + ...
          (1-post_pred_f0) .* (1 - post_pred_fP) .* KL_N1 + post_pred_f0 .* (1-post_pred_fP) .* KL_N0;

end



function kl = compute_post_pred_kl(w_sign_feedback, gamma_feedback, posterior, pr, op, X)

sf = posterior.ep_subfunctions;

m = length(posterior.p);
fa = posterior.fa;
si = posterior.si;

pr.m = m;
if ~(isfield(pr, 'kappa_prior') && pr.kappa_prior)
    pr.p_u_nat = log(pr.p_u) - log1p(-pr.p_u);
end

% EP updates

% sign feedback updates
wsign_feedbacks = [w_sign_feedback * ones(m, 1) (1:m)'];
ca_wsf = sf.compute_wsf_cavity(fa, si.w_sign_feedback, op, pr);
ti_wsf = sf.compute_wsf_tilt(ca_wsf, pr, wsign_feedbacks);
si.w_sign_feedback = sf.update_wsf_sites(si.w_sign_feedback, ca_wsf, ti_wsf, wsign_feedbacks, op);
%fa = compute_full_approximation_w(fa, si, pr); % CANNOT DO THIS AS THEN UPDATES ARE NOT INDEPENDENT BETWEEN FEATURES! Instead we'll keep track of changes here and in the w_prior separately.
%fa = compute_full_approximation_p_u(fa, si, pr); % CANNOT DO THIS EITHER; better to not update p_u at all.

% gamma feedback updates
ca_gf = sf.compute_gamma_f_lik_cavity(fa.gamma.p_nat, fa.gamma_f_p_u.p_nat, si.gamma_feedback);
ti_gf = sf.compute_gamma_f_lik_tilt(ca_gf, pr, gamma_feedback * ones(m, 1));
si.gamma_feedback = sf.update_gamma_f_lik_sites(si.gamma_feedback, ca_gf, ti_gf, op);
fa = sf.compute_full_approximation_gamma(fa, si, pr);
%fa = compute_full_approximation_p_u(fa, si, pr); % CANNOT DO THIS EITHER; better to not update p_u at all.

% prior site updates
% note: only gamma feedback update propagates here and not the sign update
% (but this is ok as an approximation).
ca_prior = sf.compute_sns_prior_cavity(fa, si.w_prior, op, pr);
ti_prior = sf.compute_sns_prior_tilt(ca_prior, op, pr);
si.w_prior = sf.update_sns_prior_sites(si.w_prior, ca_prior, ti_prior, op, pr);

% changes in parameters, account for both the gamma feedback change and the
% sign feedback change!
delta_tau = (si.w_prior.w.normal_tau - posterior.si.w_prior.w.normal_tau) + (si.w_sign_feedback.normal_tau - posterior.si.w_sign_feedback.normal_tau);
delta_mu = (si.w_prior.w.normal_mu - posterior.si.w_prior.w.normal_mu) + (si.w_sign_feedback.normal_mu - posterior.si.w_sign_feedback.normal_mu);

% KL
alpha = 1 + diag(posterior.sigma) .* delta_tau;

sigmax = posterior.sigma * X;
xTsigmax = sum(X .* sigmax, 1);
xTsigma_newx = bsxfun(@minus, xTsigmax, bsxfun(@rdivide, sigmax.^2, alpha ./ delta_tau));

if isfield(pr, 'sigma2_prior') && pr.sigma2_prior
    residual_var = 1 / posterior.fa.sigma2.imean;
else
    residual_var = pr.sigma2;
end

part1 = 0.5 * log(bsxfun(@rdivide, xTsigmax + residual_var, xTsigma_newx + residual_var));
part2_numerator = xTsigma_newx + residual_var + bsxfun(@times, sigmax, (posterior.mean .* delta_tau - delta_mu) ./ alpha).^2;
part2_denumerator = 2 * (xTsigmax + residual_var);

kl = sum(part1 + bsxfun(@rdivide, part2_numerator , part2_denumerator) - 0.5, 2);

end
