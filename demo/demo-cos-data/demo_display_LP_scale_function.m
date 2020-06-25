rng('default');
clear all;
clc;

% Setup prediction LP. We'll use a constant mean and a squared
% exponential covariance. We must use mean/covariance functions that
% support an extended GPML syntax allowing the calculation of second
% partial dervivatives with respect to hyperparameters. The
% gpml_extensions package contains implementations of some common
% choices.


% This code code had been taken from the Roman Garnett repository and
% modified by Baptiste Rubino-Moyner. 

% This implemantation shows the effect of increasing the number of
% observation on the active Laplace approximation.

%If you want to plot the selection process for the next observation go to 
%active_gp_hyperlearning/select_nextpoint and set to 1 the plot_needed
%variable

% Last modifications : 
% - change the lik from LikLaplace to likGauss
% - change the prior from laplace_prior to constant_prior and reduced the
% input from  get_prior(@constant_prior,0 ,1) to get_prior(@constant_prior,1)

% Error Status : 
% Cholesky factorization error "Matrix must be positive definite." after
% the 5th observations points 

for k=6:20 % Loop from 1 to 20 observations
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

    x_star = [linspace(0,1,100)]';

    y_star = cos_function(x_star);


    % setup problem struct
    problem.num_evaluations  = k;
    problem.candidate_x_star = x_star;

    % function is a simple lookup table
    problem.f                = ...
        @(x) (y_star(find(all(bsxfun(@eq, x, x_star), 2))));

    % actively learn GP hyperparameters
    results = learn_gp_hyperparameters(problem, model);

    [~, ~, f_star_mean, f_star_variance, log_probabilities] = ...
        gp(results.map_hyperparameters(end), model.inference_method, ...
           model.mean_function, model.covariance_function, model.likelihood, ...
           results.chosen_x, results.chosen_y, x_star, y_star);

    report = sprintf('LAPLACE active: E[log p(y* | x*, D)] = %0.3f', ...
                     mean(log_probabilities));
    fprintf('%s\n', report);

    x = results.chosen_x;
    y = results.chosen_y;

    fig = figure(1);
    set(gcf, 'color', 'white');
    plot_predictions_fonction; % Plot with adequat range for x-axis
    title(report);
    title([report,' with ',num2str(k),' obeservations']);
    legend('95%','Objective function','observation','mean')
    saveas(fig ,sprintf('FIG%d.png',k)) % save figures in the Active_gp_hyperlearning folder
end

