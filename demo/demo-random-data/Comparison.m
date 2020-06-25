rng('default');
clear all;

% Setup prediction GP. We'll use a constant mean and a squared
% exponential covariance. We must use mean/covariance functions that
% support an extended GPML syntax allowing the calculation of second
% partial dervivatives with respect to hyperparameters. The
% gpml_extensions package contains implementations of some common
% choices.

%The implemantation works well when the observation_number is setup. But
%not when we want to make a loop over observation, error -> Chol
%factorization 

observation_number = 15;
k = number_observation

model.mean_function       = {@constant_mean};
model.covariance_function = {@isotropic_sqdexp_covariance};
model.likelihood          = @likGauss;

% initial hyperparameters
offset       = 1;
length_scale = 1.25;
output_scale = 2;
noise_std    = 0.75;

true_hyperparameters.mean = offset;
true_hyperparameters.cov  = log([length_scale; output_scale]);
true_hyperparameters.lik  = log(noise_std);

% Setup hyperparameter prior. We'll use independent normal priors on
% each hyperparameter.

% N(0, 0.5^2) priors on each log covariance parameter
priors.cov  = ...
    {get_prior(@gaussian_prior, 0, 1), ...
     get_prior(@gaussian_prior, 0, 1)};

% N(0.1, 0.5^2) prior on log noise
priors.lik  = {get_prior(@gaussian_prior, 0, 1)};

% N(0, 1) prior on constant mean
priors.mean = {get_prior(@gaussian_prior, 0, 1)};

model.prior = get_prior(@independent_prior, priors);
model.inference_method = ...
    add_prior_to_inference_method(@exact_inference, model.prior);

% generate demo data

num_points = 500;

x_star_ = linspace(-5, 5, num_points)';

x_star = [x_star_];

  


mu = feval(model.mean_function{:},       true_hyperparameters.mean, x_star);
K  = feval(model.covariance_function{:}, true_hyperparameters.cov,  x_star);

K = (K + K') / 2;

y_star = mvnrnd(mu, K)';
y_star = y_star + exp(true_hyperparameters.lik) * randn(size(y_star));

% setup problem struct
problem.num_evaluations  = k;
problem.candidate_x_star = x_star;

% function is a simple lookup table
problem.f                = ...
    @(x) (y_star(find(all(bsxfun(@eq, x, x_star), 2))));

% actively learn GP hyperparameters
results = learn_gp_hyperparameters(problem, model);

[~, ~, f_star_mean_GP, f_star_variance_GP, log_probabilities_GP] = ...
    gp(results.map_hyperparameters(end), model.inference_method, ...
       model.mean_function, model.covariance_function, model.likelihood, ...
       results.chosen_x, results.chosen_y, x_star, y_star);

report = sprintf('GP active: E[log p(y* | x*, D)] = %0.3f', ...
                 mean(log_probabilities_GP));
fprintf('%s\n', report);

x = results.chosen_x;
y = results.chosen_y;

figure(1);
set(gcf, 'color', 'white');
subplot(2, 1, 1);
hold('off');

fill([x_star; flipud(x_star)], ...
     [       f_star_mean_GP + 2 * sqrt(f_star_variance_GP); ...
      flipud(f_star_mean_GP - 2 * sqrt(f_star_variance_GP))], ...
     [0.9, 0.9, 1], ...
     'edgecolor', 'none');

hold('on');

plot(x_star, y_star, 'r.');
plot(x, y, 'k+');
plot(x_star, f_star_mean_GP, '-', ...
     'color', [0, 0, 0.8]);

axis([-5, 5, -4, 6]);
set(gca, 'tickdir', 'out', ...
         'box',     'off');
title(report);

% compare to random sampling
ind = randperm(num_points, problem.num_evaluations);

x = x_star(ind, :);
y = y_star(ind);

map_hyperparameters_random = minimize_minFunc(model, x, y);

[~, ~, f_star_mean, f_star_variance, log_probabilities] = ...
    gp(map_hyperparameters_random, model.inference_method, ...
       model.mean_function, model.covariance_function, model.likelihood, ...
       x, y, x_star, y_star);

report = sprintf('GP random: E[log p(y* | x*, D)] = %0.3f', ...
                 mean(log_probabilities));
fprintf('%s\n', report);

subplot(2, 1, 2);
plot_predictions;
title(report);


%####################### LAPLACE ##########################################

rng('default');

% Setup prediction GP. We'll use a constant mean and a squared
% exponential covariance. We must use mean/covariance functions that
% support an extended GPML syntax allowing the calculation of second
% partial dervivatives with respect to hyperparameters. The
% gpml_extensions package contains implementations of some common
% choices.

model.mean_function       = {@constant_mean};
model.covariance_function = {@isotropic_sqdexp_covariance};
model.likelihood          = @likGauss;

% Setup hyperparameter prior. We'll use independent normal priors on
% each hyperparameter.

% N(0, 0.5^2) priors on each log covariance parameter
priors.cov  = ...
    {get_prior(@constant_prior, 1), ...
     get_prior(@constant_prior, 1)};

% N(0.1, 0.5^2) prior on log noise
priors.lik  = {get_prior(@constant_prior, 1)};

% N(0, 1) prior on constant mean
priors.mean = {get_prior(@constant_prior, 1)};

model.prior = get_prior(@independent_prior, priors);
model.inference_method = ...
    add_prior_to_inference_method(@exact_inference, model.prior);

% generate demo data

num_points = 500;

x_star_ = linspace(-5, 5, num_points)';

x_star = [x_star_];

mu = feval(model.mean_function{:},       true_hyperparameters.mean, x_star);
K  = feval(model.covariance_function{:}, true_hyperparameters.cov,  x_star);

K = (K + K') / 2;

y_star = mvnrnd(mu, K)';
y_star = y_star + exp(true_hyperparameters.lik) * randn(size(y_star));

% setup problem struct
problem.num_evaluations  = k;
problem.candidate_x_star = x_star;

% function is a simple lookup table
problem.f                = ...
    @(x) (y_star(find(all(bsxfun(@eq, x, x_star), 2))));

% actively learn LP hyperparameters
results = learn_gp_hyperparameters(problem, model);

[~, ~, f_star_mean_LP, f_star_variance_LP, log_probabilities_LP] = ...
    gp(results.map_hyperparameters(end), model.inference_method, ...
       model.mean_function, model.covariance_function, model.likelihood, ...
       results.chosen_x, results.chosen_y, x_star, y_star);

report = sprintf('LAPLACE active: E[log p(y* | x*, D)] = %0.3f', ...
                 mean(log_probabilities_LP));
fprintf('%s\n', report);

x = results.chosen_x;
y = results.chosen_y;

figure(2);
set(gcf, 'color', 'white');
subplot(2, 1, 1);

hold('off');

fill([x_star; flipud(x_star)], ...
     [       f_star_mean_LP + 2 * sqrt(f_star_variance_LP); ...
      flipud(f_star_mean_LP - 2 * sqrt(f_star_variance_LP))], ...
     [0.9, 0.9, 1], ...
     'edgecolor', 'none');

hold('on');

plot(x_star, y_star, 'r.');
plot(x, y, 'k+');
plot(x_star, f_star_mean_LP, '-', ...
     'color', [0, 0, 0.8]);

axis([-5, 5, -4, 6]);
set(gca, 'tickdir', 'out', ...
         'box',     'off');

title(report);

% compare to random sampling
ind = randperm(num_points, problem.num_evaluations);

x = x_star(ind, :);
y = y_star(ind);

map_hyperparameters_random = minimize_minFunc(model, x, y);

[~, ~, f_star_mean, f_star_variance, log_probabilities] = ...
    gp(map_hyperparameters_random, model.inference_method, ...
       model.mean_function, model.covariance_function, model.likelihood, ...
       x, y, x_star, y_star);

report = sprintf('LAPLACE random: E[log p(y* | x*, D)] = %0.3f', ...
                 mean(log_probabilities));
fprintf('%s\n', report);

subplot(2, 1, 2);
plot_predictions;
title(report);







fig = figure(3);

hold('off');

fill([x_star; flipud(x_star)], ...
     [       f_star_mean_GP + 2 * sqrt(f_star_variance_GP); ...
      flipud(f_star_mean_GP - 2 * sqrt(f_star_variance_GP))], ...
     [0.9, 0.9, 1], ...
     'edgecolor', 'b');
 
 hold('on');
 
 h = fill([x_star; flipud(x_star)], ...
     [       f_star_mean_LP + 2 * sqrt(f_star_variance_LP); ...
      flipud(f_star_mean_LP - 2 * sqrt(f_star_variance_LP))], ...
     [1, 0.9, 0.9], ...
     'edgecolor', 'none');

hold('on');

plot(x_star, y_star, 'r.');
plot(x, y, 'k+');
plot(x_star, f_star_mean_LP, '-', ...
     'color', [0.8, 0, 0]);
hold('on');
plot(x_star, f_star_mean_GP, '-', ...
     'color', [0.3, 0.3, 1]);

axis([-5, 5, -4, 6]);
set(gca, 'tickdir', 'out', ...
         'box',     'off');
[~,h_legend] = legend('GP','Laplace');
PatchInLegend = findobj(h_legend, 'type', 'patch');
set(h,'facealpha',.2)
title(['Comparison of LP and GP active D = 1 with ', num2str(k),' observation(s)'])
saveas(fig ,sprintf('FIG%d.png',k))
 