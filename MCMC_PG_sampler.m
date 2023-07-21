classdef MCMC_PG_sampler < handle
    properties
        %% general properties 
        Nt;             % number of theta samples (length of chain)
        Nx;             % number of particles
        m;              % model
        LL;             % log-likelihood estimates
        tsamples;       % parameter (theta) samples
        Sig;            % covariance matrix for random walk
        ini;            % initial value    
        
        Xk;
         
        %% other
        ct;             % computation time
        loglike_scale;  % use to compare with SMC, default = 1
        acc;
    end
    
    methods
       % constructor
       function obj = MCMC_PG_sampler(m, Sig, ini, varargin) 
           p = inputParser;
           addParameter(p, 'loglike_scale', 1);
           parse(p, varargin{:});
            
           obj.loglike_scale = p.Results.loglike_scale;
           obj.m = m;
           obj.Sig = Sig;
           obj.ini = ini;
       end
       
       %% particle marginal Metropolis-Hastings sampler
       function sample(o, Nt, Nx, MALA_stepsize)
           o.Nt = Nt;
           o.Nx = Nx;
           
           tic;
           
           % initialise
           o.tsamples = zeros(Nt, o.m.np);
           o.tsamples(1, :) = o.ini;
           
           % initial state trajectory
            o.Xk = reshape(o.m.x, [1, o.m.T]);
            
            o.acc = 0;
 
           for i = 2:Nt
               % print progress
               if mod(i, ceil(o.Nt/10)) == 0
                   sprintf('Completed: %d%%', round(i/Nt*100))
               end
               
               o.tsamples(i, :) = o.tsamples(i-1, :);
               
               % draw new path
                [~, Xm, logWt] = ParticleFilter.conditional(o.m, o.tsamples(i, :), o.Nx, o.Xk);
                o.Xk = ParticleFilter.drawB(o.m, Xm, logWt, o.tsamples(i, :));
                
               % current posterior
               log_posterior = o.m.prior_lpdf(o.tsamples(i, :)); 
               log_posterior = log_posterior + sum(log_state_density(o.m, o.tsamples(i, :), o.Xk), 2);
               log_posterior = log_posterior + log_obs_density(o.m, o.m.T, o.tsamples(i, :), o.Xk);
               
               epsilon_sq = MALA_stepsize*o.Sig;
               grad_current = o.m.grad_post(o.m.T, o.Xk, o.tsamples(i, :), 1); 
               mu_new = o.tsamples(i, :) + 0.5*grad_current*epsilon_sq;
               
               % propose new
               theta_new = mvnrnd(mu_new, epsilon_sq); 
               
               log_posterior_new = o.m.prior_lpdf(theta_new); 
               log_posterior_new = log_posterior_new + sum(log_state_density(o.m, theta_new, o.Xk), 2);
               log_posterior_new = log_posterior_new + sum(log_obs_density(o.m, o.m.T, theta_new, o.Xk), 2);
               
               % proposal posterior
                grad_new = o.m.grad_post(o.m.T, o.Xk, theta_new, 1);
                mu_current = theta_new + 0.5*grad_new*epsilon_sq;

               % Metropolis-Hastings ratio
               prop = logmvnpdf(o.tsamples(i, :), mu_current, epsilon_sq) - logmvnpdf(theta_new, mu_new, epsilon_sq);
               MHRatio = exp(log_posterior_new - log_posterior + prop);

               % accept/reject 
               if (rand < MHRatio) 
                   o.tsamples(i, :) = theta_new;
                   o.acc = o.acc + 1;
               end
            end

           o.ct = toc;
           o.acc = o.acc/Nt;
       end % PMMH

    end
    
end