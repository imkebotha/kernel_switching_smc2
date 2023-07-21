classdef MCMC_sampler < handle
    properties
        %% general properties 
        Nt;             % number of theta samples (length of chain)
        Nx;             % number of particles
        m;              % model
        LL;             % log-likelihood estimates
        tsamples;       % parameter (theta) samples
        Sig;            % covariance matrix for random walk
        ini;            % initial value         
         
        %% other
        ct;             % computation time
        loglike_scale;  % use to compare with SMC, default = 1
    end
    
    methods
       % constructor
       function obj = MCMC_sampler(m, Sig, ini, varargin) 
           p = inputParser;
           addParameter(p, 'loglike_scale', 1);
           parse(p, varargin{:});
            
           obj.loglike_scale = p.Results.loglike_scale;
           obj.m = m;
           obj.Sig = Sig;
           obj.ini = ini;
       end
       
       %% particle marginal Metropolis-Hastings sampler
       function sample(o, Nt, Nx)
           o.Nt = Nt;
           o.Nx = Nx;
           
           % preallocate
           o.LL = zeros(Nt, 1);
           o.tsamples = zeros(Nt, o.m.np);
           
           % initialise
           tic;
           acc = 0;
           o.tsamples(1, :) = o.ini;

           o.LL(1) = ParticleFilter.standard(o.m, o.tsamples(1, :), Nx);
           log_posterior = o.loglike_scale*o.LL(1) + o.m.prior_lpdf(o.tsamples(1, :));
           
           for i = 1:Nt
               % print progress
               if mod(i, round(Nt/10)) == 0
                   sprintf('Completed: %d%%', round(i/Nt*100))
				   save('temp.mat', 'o');
               end

               % update theta
               % propose new
               theta_new = mvnrnd(o.tsamples(i, :), o.Sig);

               loglike_new = ParticleFilter.standard(o.m, theta_new, Nx);
               log_posterior_new = o.loglike_scale*loglike_new + o.m.prior_lpdf(theta_new);

               % Metropolis-Hastings ratio
               MHRatio = exp(log_posterior_new - log_posterior);

               % accept/reject 
               if (rand < MHRatio) 
                   o.tsamples(i+1, :) = theta_new;
                   o.LL(i+1) = loglike_new;
                   log_posterior = log_posterior_new;
                   acc = acc + 1;
               else
                   o.tsamples(i+1, :) = o.tsamples(i, :);
                   o.LL(i+1) = o.LL(i);
               end
            end

            o.ct = toc;
           
       end % PMMH

    end
    
end