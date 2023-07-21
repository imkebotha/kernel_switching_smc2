function plot_samples(m, samples, varargin)

transform = true;

% parse input 
p = inputParser;
addParameter(p, 'TracePlots', false);
addParameter(p, 'Weights', ones(size(samples, 1), 1));
addParameter(p, 'Reference', [])
addParameter(p, 'Parameters', []);
parse(p, varargin{:});  

W = p.Results.Weights./sum(p.Results.Weights);
if transform
    samples = m.transform(samples, true); %
end

if ~isempty(p.Results.Parameters)
    ind = p.Results.Parameters;
else
    ind = 1:m.np;
end
nump = length(ind);

plot_reference = ~isempty(p.Results.Reference);
if plot_reference
    pmcmcobj = p.Results.Reference;
    if transform
        mcmc_samples = m.transform(pmcmcobj.tsamples, true);
    else
        mcmc_samples = pmcmcobj.tsamples;
    end
end

% plot prior
prior_samples = m.prior_rnd(10000);
have_tv = all(~isnan(m.theta));
if have_tv
    if transform
        prior_samples = m.transform(prior_samples, true);
        tv = m.transform(m.theta, true);
    else
        tv = m.theta;
    end
end


x_limits = zeros(nump, 2);

figure; tiledlayout('flow');
num_subplots = 0;
for k = 1:nump
    i = ind(k);
    nexttile
    hold on;
    if iscell(samples)
        for j = 1:length(samples)
            [y, x] = ksdensity(samples{j}(:, i), 'Weights', W); 
            plot(x, y, 'LineWidth', 1)
        end
    else
        [y, x] = ksdensity(samples(:, i), 'Weights', W); 
        plot(x, y, 'LineWidth', 1)
    end
    
    x_min = min(x);
    x_max = max(x);
    
    x_min = x_min - 0.9*min(0.1, abs(x_min)); 
    x_max = x_max + 1.1*min(0.1, abs(x_max)); 
         
    x_limits(i, :) = [x_min x_max];

    if plot_reference
        [y, x] = ksdensity(mcmc_samples(:, i));
        plot(x, y, 'LineWidth', 1, 'Color', 'k')
    end
    title(m.names(i), 'FontSize', 14)

    if have_tv; xline(tv(i)); end
    [y, x] = ksdensity(prior_samples(:, i));
    plot(x, y, 'LineWidth', 1)
    xlim(x_limits(i, :));
    hold off
    
    num_subplots = num_subplots + 1;
    if (num_subplots == 12 && i ~= m.np) 
        num_subplots = 0;
        figure; tiledlayout('flow');
    end
end

%% trace plots

if p.Results.TracePlots
    figure; tiledlayout('flow');
    num_subplots = 0;

    for k = 1:nump
        i = ind(k);
        nexttile
        plot(samples(:, i));
        title(m.names(i), 'FontSize', 14)
        num_subplots = num_subplots + 1;
        if (num_subplots == 12 && i ~= m.np) 
            num_subplots = 0;
            figure; tiledlayout('flow');
        end
    end
end

end