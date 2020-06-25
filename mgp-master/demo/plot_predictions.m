hold('off');

fill([x_star; flipud(x_star)], ...
     [       f_star_mean + 2 * sqrt(f_star_variance); ...
      flipud(f_star_mean - 2 * sqrt(f_star_variance))], ...
     [0.9, 0.9, 1], ...
     'edgecolor', 'none');

hold('on');

plot(x_star, y_star, 'r.');
plot(x, y, 'k+');
plot(x_star, f_star_mean, '-', ...
     'color', [0.8, 0, 0]);

axis([-3, 3, -4, 6]);
set(gca, 'tickdir', 'out', ...
         'box',     'off');
