% generate demo regression data from GPML toolkit
mean_function       = {@meanSum, {@meanLinear, @meanConst}};
covariance_function = {@covMaterniso, 3};

slope        = 0.5;
offset       = 1;
length_scale = 0.25;
output_scale = 1;
noise_std    = 0.1;

true_hyperparameters.mean = [slope; offset];
true_hyperparameters.cov  = log([length_scale; output_scale]);
true_hyperparameters.lik  = log(noise_std);

% generate training data
num_train = 20;

x = gpml_randn(0.3, num_train, 1);

mu = feval(mean_function{:},       true_hyperparameters.mean, x);
K  = feval(covariance_function{:}, true_hyperparameters.cov,  x);

y = mu + chol(K)' * gpml_randn(0.15, num_train, 1) + ...
    exp(true_hyperparameters.lik) * gpml_randn(0.2, num_train, 1);

% generate test data
num_test = 100;

x_star = linspace(-3, 3, num_test)';

[y_star_mean, y_star_variance] = gp(true_hyperparameters, [], ...
        mean_function, covariance_function, [], x, y, x_star);

y_star = y_star_mean + ...
         sqrt(y_star_variance) .* gpml_randn(0.5, num_test, 1);
%##########################################################################

% setup prediction GP

% We'll use a constant mean and a squared exponential covariance. We
% must use mean/covariance functions that support an extended GPML
% syntax allowing the calculation of second partial dervivatives with
% respect to hyperparameters. The gpml_extensions package contains
% implementations of some common choices.

inference_method    = @exact_inference;
mean_function       = {@constant_mean};
covariance_function = {@isotropic_sqdexp_covariance};

% initial hyperparameters
offset       = 1;
length_scale = 1;
output_scale = 1;
noise_std    = 0.05;

hyperparameters.mean = offset;
hyperparameters.cov  = log([length_scale; output_scale]);
hyperparameters.lik  = log(noise_std);

% find MLE hyperparameters
mle_hyperparameters = minimize(hyperparameters, @gp, 50, inference_method, ...
        mean_function, covariance_function, [], x, y);

% get predictions from GP conditioned on MLE hyperparameters
[~, ~, f_star_mean_mle, f_star_variance_mle, log_probabilities_mle] = ...
    gp(...
    mle_hyperparameters,...
    inference_method, ...
    mean_function, ...
       covariance_function, [], x, y, x_star, y_star);

report = sprintf(' GP/MLE: E[log p(y* | x*, D)] = %0.3f', ...
                 mean(log_probabilities_mle));
fprintf('%s\n', report);

% plot GP predictions
figure(1);
set(gcf, 'color', 'white');

% -------------------------------------------------------------------------

% add hyperparameter priors

% N(0, 0.2) priors on each log covariance parameter
priors.cov  = ...
    {get_prior(@gaussian_prior, log(1), 0.2), ...
     get_prior(@gaussian_prior, log(1), 0.2)};

% N(0.1, 0.2^2) prior on log noise
priors.lik  = ...
    {get_prior(@gaussian_prior, log(0.1), 0.2)};

% N(0, 0.5^2) prior on constant mean
priors.mean = ...
    {get_prior(@gaussian_prior, 0, 0.5^2)};

prior = get_prior(@independent_prior, priors);
inference_method = {@inference_with_prior, inference_method, prior};

% find MAP hyperparameters
map_hyperparameters = minimize(hyperparameters, @gp, 50, inference_method, ...
        mean_function, covariance_function, [], x, y);

% get predictions from GP conditioned on MAP hyperparameters
[~, ~, f_star_mean_map, f_star_variance_map, log_probabilities_map] = ...
    gp(map_hyperparameters, inference_method, mean_function, ...
       covariance_function, [], x, y, x_star, y_star);

report = sprintf(' GP/MAP: E[log p(y* | x*, D)] = %0.3f', ...
                 mean(log_probabilities_map));
fprintf('%s\n', report);

% plot GP predictions




% -------------------------------------------------------------------------
% find Laplace hyperparameters
% add hyperparameter priors

% N(0, 0.2) priors on each log covariance parameter
priors.cov  = ...
    {get_prior(@laplace_prior, log(1), 0.2), ...
     get_prior(@laplace_prior, log(1), 0.2)};

% N(0.1, 0.2^2) prior on log noise
priors.lik  = ...
    {get_prior(@laplace_prior, log(0.1), 0.2)};

% N(0, 0.5^2) prior on constant mean
priors.mean = ...
    {get_prior(@laplace_prior, 0, 0.5^2)};

prior = get_prior(@independent_prior, priors);


inference_method    = @laplace_inference;
inference_method = {@inference_with_prior, inference_method, prior};

Laplace_hyperparameters = minimize(hyperparameters,...
    @gp, 50,...
    inference_method, ...
        mean_function, covariance_function, [], x, y);



[~, ~, f_star_mean_lp, f_star_variance_lp, log_probabilities_lp] = ...
    gp(Laplace_hyperparameters,...
inference_method,...
    mean_function, ...
        covariance_function, [], x, y, x_star, y_star);

report = sprintf('GP/Laplace: E[log p(y* | x*, D)] = %0.3f', ...
                 mean(log_probabilities_lp));
fprintf('%s\n', report);

% plot  GP with Laplace approx. predictions

%############################# PLOT ##################################


hold('off');
figure(1);

fill([x_star; flipud(x_star)], ...
     [       f_star_mean_mle + 2 * sqrt(f_star_variance_mle); ...
      flipud(f_star_mean_mle - 2 * sqrt(f_star_variance_mle))], ...
     [0.9, 0.9, 1], ...
     'edgecolor', 'b');
 hold('on');
 h = fill([x_star; flipud(x_star)], ...
     [       f_star_mean_lp + 2 * sqrt(f_star_variance_lp); ...
      flipud(f_star_mean_lp - 2 * sqrt(f_star_variance_lp))], ...
     [1, 0.9, 0.9], ...
     'edgecolor', 'r');
 
hold('on');
plot(x_star, y_star, 'r.');
plot(x, y, 'k+');
plot(x_star, f_star_mean, '-', ...
     'color', [0, 0, 0]);

axis([-3, 3, -4, 6]);
set(gca, 'tickdir', 'out', ...
         'box',     'off'); 
title('Comparison between MLE and Laplace Approx')
[~,h_legend] = legend('MLE','Laplace');
PatchInLegend = findobj(h_legend, 'type', 'patch');
set(h,'facealpha',.5)
hold('off');    

figure(2); 

fill([x_star; flipud(x_star)], ...
     [       f_star_mean_map + 2 * sqrt(f_star_variance_map); ...
      flipud(f_star_mean_map - 2 * sqrt(f_star_variance_map))], ...
     [0.3, 0.7, 0.3], ...
     'edgecolor', 'g');
 hold('on');
 h = fill([x_star; flipud(x_star)], ...
     [       f_star_mean_lp + 2 * sqrt(f_star_variance_lp); ...
      flipud(f_star_mean_lp - 2 * sqrt(f_star_variance_lp))], ...
     [1, 0.9, 0.9], ...
     'edgecolor', 'r');
 
hold('on');
plot(x_star, y_star, 'r.');
plot(x, y, 'k+');
plot(x_star, f_star_mean, '-', ...
     'color', [0, 0, 0]);

axis([-3, 3, -4, 6]);
set(gca, 'tickdir', 'out', ...
         'box',     'off');
title('Comparison between MAP and Laplace Approx')
[~,h_legend] = legend('MAP','Laplace');
PatchInLegend = findobj(h_legend, 'type', 'patch');
set(h,'facealpha',.5)


 





