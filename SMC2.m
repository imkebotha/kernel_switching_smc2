classdef SMC2 < handle
    %% Visible properties
    properties
        % constants -------------------------------------------------------
        m;                  % state-space model object
        Nt;                 % number of parameter (theta) particles
        Nx_PMMH;            % number of state particles for PMMH
        Nx_PG;
        
        % efficiency targets ----------------------------------------------
        targetSJD;          % target min squared jumping distance 
        targetESS;          % target ESS (as a fraction of Nt)
        MALA_ar_target;     % target acceptance rate for MALA
        PMMH_ar_min_target; % min acceptance rate for PMMH
        
        % settings --------------------------------------------------------
        dataAnnealing;      % true for data annealing, false for density tempering
        kernelAdaptation;   % 3 options: "none", "always", "lag" 
        fixed_kernel;       % kernel used to reweight
        switch_kernel;      % alternative kernel for mutation
        
        % results ---------------------------------------------------------
        results_summary;    % table with mutation step related results
        target_means;       % sample means at each iteration
        targetSJD_hist;     % target min squared jumping distance 
        ct;                 % computation time
        TLL;                % number of log-likelihood calculations * number of observations
        
        % samples ---------------------------------------------------------
        tsamples;           % parameter samples
        Xk;                 % invariant path
    end
     
    %% Hidden properties
    properties (Hidden)
        LL;                 % log-likelihood estimates
        xsamples;           % state particles
        tSig;               % covariance of particle set
        PMMH_stepsize;      % scaling factor for PMMH proposal
        MALA_stepsize;      % scaling factor for PG proposal
        MALA_K;             % number of MALA iterations
        test_K;             % number of mutations used to test a particular kernel
        Nx;                 % number of state particles
        logNWt;             % parameter normalised log weights 
        mutationKernel;     % current mutation kernel
        t;                  % current number of observations
        g;                  % current temperature
        temperature_prev;   % previous temperature
    end
    
    %% Visible methods
    methods
        % -----------------------------------------------------------------
        % Constructor
        % -----------------------------------------------------------------
        function o = SMC2(m, method, varargin)
            % parse optional input
            p = inputParser;
            addParameter(p, 'targetESS', 0.5);
            addParameter(p, 'kernel', 'pmmh');
            addParameter(p, 'kernelAdaptation', 'none');
            addParameter(p, 'MALA_ar_target', 0.574);
            parse(p, varargin{:});
            
            % set values
            o.targetSJD = 4;
            o.targetESS = p.Results.targetESS;
            o.MALA_ar_target = p.Results.MALA_ar_target;
            o.PMMH_ar_min_target = 0.07; % don't want the acceptance rate of PMMH to drop below 7%
            o.m = m;

            o.dataAnnealing = method == "DataAnnealing";
            o.mutationKernel = lower(p.Results.kernel);
            o.fixed_kernel = o.mutationKernel;
            o.kernelAdaptation = p.Results.kernelAdaptation;
            o.MALA_K = 5;
            o.test_K = 5;
            
            % create results table
            varNames = ["NumObs", "Temperature", "R_PMMH", "R_PG", "AR_PMMH", "AR_PG_1", "AR_PG_2", "Pen_PMMH", "Pen_PG", "AR", "SJD"];
            varTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
            o.results_summary = table('Size', [0, length(varNames)], 'VariableTypes', varTypes, 'VariableNames', varNames);
            
            
            if o.mutationKernel == "pmmh"
                o.switch_kernel = "pg";
            else
                o.switch_kernel = "pmmh";
            end
            
            if o.kernelAdaptation ~= "none" && ~o.dataAnnealing && o.fixed_kernel == "pmmh"
                error("Cannot switch between standard PMMH and PG when using density tempering SMC");
            end
        end % EASMC
        
        % -----------------------------------------------------------------
        % Sampler
        % -----------------------------------------------------------------
        function sample(o, Nt, Nx, r)
            o.Nt = Nt; 
            % default value of r is 1
            if nargin < 4
                r = 1;
            end
            o.Nx_PMMH = Nx;
            o.Nx_PG = ceil(o.Nx_PMMH * r);
            if o.mutationKernel == "pg"
                o.Nx = o.Nx_PG;
            else
                o.Nx = o.Nx_PMMH;
            end
            o.resetProperties();
            tic();
            
            % initialise 
            o.tsamples = o.m.prior_rnd(o.Nt);
            o.tSig = cov(o.tsamples);
            o.logNWt = -log(o.Nt)*ones(o.Nt, 1); 
            
            if ~o.dataAnnealing && o.mutationKernel == "pg"
                o.Xk = zeros(o.Nt, o.m.T);
                o.Xk(:, 1, :) = o.m.x1_rnd(o.tsamples); 
                for j = 2:o.t
                    o.Xk(:, j, :) = x_rnd(o.m, o.Xk(:, j-1, :), o.tsamples, j-1, j);
                end
            elseif ~o.dataAnnealing && o.mutationKernel == "pmmh"  
                [o.LL, o.xsamples] = o.loglikelihood(o.tsamples, o.Nx);
            end
            
            past_final = o.goToNextDistribution();
            while ~past_final
                o.reweight();
                
                % re-sample and mutate
                resample_move = (o.t == o.m.T) || (exp(-logsumexp(2*o.logNWt)) < o.targetESS*o.Nt);
                if resample_move
                    o.adaptProposals();
                    o.resample(); 
                    o.adaptiveMutation();
                end
                past_final = o.goToNextDistribution();
            end
            
            % results
            o.ct = [o.ct toc];
            
            % clear properties
            o.LL = []; o.xsamples = [];  
        end % sample
    end % public methods
    
    
    methods (Hidden)
        %% Properties
        
        % reset all sampler properties for a new run
        function resetProperties(o)
            o.logNWt = [];
            o.LL = zeros(o.Nt, 1);
            o.TLL = 0;
            o.tsamples = [];
            o.xsamples = [];
            o.target_means = [];
            o.targetSJD_hist = [];
            o.PMMH_stepsize = 1;
            o.MALA_stepsize = repmat(0.01/o.m.np, 2, 1);

            if o.dataAnnealing
                o.g = 1; o.t = 0;
            else
                o.g = 0; o.t = o.m.T;
            end
        end % resetProperties
        
        %% Likelihood and posterior

        % Estimate log-likelihood 
        function [ll, x] = loglikelihood(o, theta, nx)
            x = []; 
            if o.dataAnnealing
                [ll, x] = ParticleFilter.standard(o.m, theta, nx, 'T', o.t);
                
            else % density tempering
                if o.fixed_kernel == "pmmh" 
                    ll = ParticleFilter.standard(o.m, theta, nx); % PMMH version 1
                else 
                    ll = ParticleFilter.standard(o.m, theta, nx, 'g', o.g); % PMMH version 2 
                end
            end
            o.TLL = o.TLL + size(theta, 1)*nx*o.t;
        end % loglikelihood
         
        
        % Calculate the log-posterior
        function f = logposterior(o, varargin) 
            p = inputParser;
            addParameter(p, 'LL', o.LL);
            addParameter(p, 'tsamples', o.tsamples);
            addParameter(p, 'temperature', o.g);
            parse(p, varargin{:});

            % prior
            f = sum(o.m.prior_lpdf(p.Results.tsamples), 2); 
            
            % transition density
            if o.mutationKernel == "pg"
                f = f + sum(log_state_density(o.m, p.Results.tsamples, o.Xk), 2);
            end
            
            % observation density/likelihood
            % Data Annealing with PMMH
            if o.dataAnnealing && o.mutationKernel == "pmmh"
                f = f + p.Results.LL;
                
            % Data Annealing with PG
            elseif o.dataAnnealing && o.mutationKernel == "pg"
                f = f + sum(log_obs_density(o.m, o.t, p.Results.tsamples, o.Xk), 2);
                
            % Density Tempering with PMMH version 1
            elseif ~o.dataAnnealing && o.fixed_kernel == "pmmh"
                f = f + p.Results.temperature*p.Results.LL;
                
            % Density Tempering with PMMH version 2 (if using PG to
            % reweight)
            elseif ~o.dataAnnealing && o.mutationKernel == "pmmh"
                f = f + p.Results.LL;
                
            % Density Tempering with PG
            elseif ~o.dataAnnealing && o.mutationKernel == "pg"
                f = f + p.Results.temperature*(sum(log_obs_density(o.m, o.m.T, p.Results.tsamples, o.Xk), 2));
            end
        end  % logposterior


        %% Next distribution
        
        % Set parameters for the next distribution
        function past_final = goToNextDistribution(o)
            past_final = false;
            if o.dataAnnealing && (o.t + 1) <= o.m.T
                o.t = o.t + 1;
                fprintf('Completing iteration: %d of %d\n', o.t, o.m.T); 
            elseif ~o.dataAnnealing && (o.g < 1) 
                o.temperature_prev = o.g;
                o.g = o.setNewTemperature(); 
                fprintf('Current temperature: %0.4f\n', o.g);
            else
                past_final = true;
            end
        end % goToNextDistribution
        
        
        % Determine new temperature using the bisection method
        function newTemp = setNewTemperature(o)
            busy = true;
            a = o.g; b = 1; % initial bounds 
            
            % target effective sample size
            tESS = o.targetESS*o.Nt;

            ESSa_diff = exp(-logsumexp(2*o.logNWt)) - tESS; 
            if ESSa_diff < 0
                busy = false;
                p = min(b, a + 0.005);
            end
            
            lpostg = o.logposterior();
            while (busy)

                % calculate ESS for temperature p
                p = (a + b)/2;

                lpostp = o.logposterior('temperature', p);
                logw = o.logNWt + lpostp - lpostg;
                logw(isnan(logw)) = -inf;
                lognw = logw - logsumexp(logw); 
                ESSp_diff = exp(-logsumexp(2*lognw)) - tESS; 

                % update bounds
                if ESSa_diff*ESSp_diff < 0
                    b = p;
                else
                    a = p;
                    ESSa_diff = ESSp_diff;
                end

                % set new values
                busy = abs(ESSp_diff) > 1e-2;
                if (b-a) <= 1e-4
                    p = b;
                    busy = false;
                end
            end
            newTemp = p;
        end % setNewTemperature
        
        
        %% Reweight
        
        function reweight(o)
            if o.dataAnnealing
                % data annealing with PMMH
                if o.mutationKernel == "pmmh"
                    logpost_previous = o.logposterior();
                    [ll_increment, o.xsamples] = ...
                            ParticleFilter.iteration(o.m, o.tsamples, o.Nx, o.t, o.xsamples);
                    o.LL = o.LL + ll_increment;
                    o.TLL = o.TLL + o.Nt*o.Nx*1;
                    logWt = o.logNWt + o.logposterior() - logpost_previous; 
                % data annealing with PG
                else
                    if o.t == 1
                        o.Xk = o.m.x1_rnd(o.tsamples); 
                    else
                        o.Xk = [o.Xk x_rnd(o.m, o.Xk(:, end, :), o.tsamples, o.t-1, o.t)];
                    end
                    logWt = o.logNWt + o.m.y_lpdf(o.t, o.Xk(:, o.t, :), o.tsamples);
                end
            % density tempering with PMMH or PG
            else
                logWt = o.logNWt + o.logposterior() - o.logposterior('temperature', o.temperature_prev); %o.temperatures(end-1)
            end
            logWt(isnan(logWt)) = -inf;
            o.logNWt = logWt - logsumexp(logWt); 
            o.logNWt(isnan(o.logNWt)) = -inf;
        end % reweight
    
        %% Resample
        
        function resample(o)

            % multinomial resampling
            I = randsample((1:o.Nt)', o.Nt, true, exp(o.logNWt)); 
            
            o.tsamples = o.tsamples(I, :);
            o.logNWt = -log(o.Nt)*ones(o.Nt, 1);

            if o.mutationKernel == "pmmh"
                o.LL = o.LL(I);
                if o.dataAnnealing
                    o.xsamples = o.xsamples(I, :, :);
                end
            else
                o.Xk = o.Xk(I, :, :);
            end
        end % resample

        %% Mutation

        function adaptProposals(o)
            % PMMH stepsize
            if ~isempty(o.results_summary.AR_PMMH)
                last_pmmh_ar = o.results_summary.AR_PMMH(end);
                r = last_pmmh_ar/o.PMMH_ar_min_target;
                o.PMMH_stepsize = min(1, o.PMMH_stepsize * exp(2*r-2));
            end

            % MALA stepsizes
            if ~isempty(o.results_summary.AR_PG_1)
                last_mala_ar = o.results_summary.AR_PG_1(end);
                r = last_mala_ar/o.MALA_ar_target; 
                o.MALA_stepsize(1) = o.MALA_stepsize(1) * exp(2*r-2);
            end

            if ~isempty(o.results_summary.AR_PG_2)
                last_mala_ar = o.results_summary.AR_PG_2(end);
                r = last_mala_ar/o.MALA_ar_target; 
                o.MALA_stepsize(2) = o.MALA_stepsize(2) * exp(2*r-2);
            end

            % covariance matrix
            w = exp(o.logNWt);
            mu = sum(w .* o.tsamples);
            sigma_sq = o.Nt/(o.Nt - 1) * (w.*(o.tsamples-mu))'*(o.tsamples-mu);
            o.tSig = topdm(sigma_sq);

            % SJD target
            o.targetSJD = 4*sum(mean(((o.tsamples-mu)*o.tSig^(-0.5)).^2, 2).*w); 
            o.targetSJD_hist = [o.targetSJD_hist o.targetSJD];
        end

        % Particle marginal Metropolis Hastings update for static parameters
        function PMMH_mutation_kernel(o)
            % current
            log_posterior = o.logposterior();
            
            % proposal
            theta_new = mvnrnd(o.tsamples, o.PMMH_stepsize*o.tSig); 
            [LL_new, Xm_new] = o.loglikelihood(theta_new, o.Nx);
            log_posterior_new = o.logposterior('LL', LL_new, 'tsamples', theta_new);
            
            % Metropolis-Hastings ratio
            MHRatio = exp(log_posterior_new - log_posterior);
			MHRatio(isnan(MHRatio)) = 0;
            I = rand(o.Nt, 1) < MHRatio;

            % update values
            o.tsamples(I, :) = theta_new(I, :);
            o.LL(I) = LL_new(I);

            if o.dataAnnealing
                o.xsamples(I, :, :) = Xm_new(I, :, :); 
            end
            
            % update acceptance rate
            o.results_summary.AR_PMMH(end) = o.results_summary.AR_PMMH(end) + mean(I);
        end % PMMH_mutation_kernel
        
        
        % Particle Gibbs update for the latent state trajectories
        function drawLatentStates(o, nx)
            [o.LL, Xm, logwt] = ParticleFilter.conditional(o.m, o.tsamples, nx, o.Xk, 'T', o.t, 'g', o.g); 
            o.Xk = ParticleFilter.drawB(o.m, Xm, logwt, o.tsamples);
            o.TLL = o.TLL + o.Nt*nx*o.t;
        end % drawLatentStates

        
        % MALA update for the parameters in parameter_block i
        function MALA_update(o, i)
            b = o.m.parameter_blocks{i};
            epsilon_sq = o.MALA_stepsize(i)*o.tSig(b, b);

            % current
            tcurrent = o.tsamples;
            log_posterior = o.logposterior();
            grad_current = o.m.grad_post(o.t, o.Xk(:, 1:o.t, :), tcurrent, o.g); 
            mu_new = tcurrent(:, b) + 0.5*grad_current(:, b)*epsilon_sq;

            % proposal
            tnew = o.tsamples;
            tnew(:, b) = mvnrnd(mu_new, epsilon_sq); 

            % proposal posterior
            grad_new = o.m.grad_post(o.t, o.Xk(:, 1:o.t, :), tnew, o.g);
            mu_current = tnew(:, b) + 0.5*grad_new(:, b)*epsilon_sq;
            log_posterior_new = o.logposterior('tsamples', tnew);

            % proposal ratio
            prop = logmvnpdf(tcurrent(:, b), mu_current, epsilon_sq) - logmvnpdf(tnew(:, b), mu_new, epsilon_sq);
            MHRatio = exp(log_posterior_new - log_posterior + prop);
            MHRatio(isnan(MHRatio)) = 0;
            I = rand(o.Nt, 1) < MHRatio; 
            o.tsamples(I, :) = tnew(I, :);

            % update acceptance rate
            if i == 1
                o.results_summary.AR_PG_1(end) = o.results_summary.AR_PG_1(end) + mean(I);
            else
                o.results_summary.AR_PG_2(end) = o.results_summary.AR_PG_2(end) + mean(I);
            end
            
        end % MALA_update
        
        
        % single mutation function for SMC^2
        function Mutation(o) 
            if o.mutationKernel == "pmmh"
                PMMH_mutation_kernel(o);
                o.results_summary.R_PMMH(end) = o.results_summary.R_PMMH(end) + 1;
            else
                drawLatentStates(o, o.Nx);
                % conditional on the invariant path, do K MALA iterations
                for k = 1:o.MALA_K
                    MALA_update(o, 1);
                    MALA_update(o, 2);
                end
                o.results_summary.R_PG(end) = o.results_summary.R_PG(end) + 1;
            end
        end % Mutation
        
        
        % full adaptive mutation step for SMC^2
        function adaptiveMutation(o)
            % initialise --------------------------------------------------
            dtheta = o.tsamples;
            
            % add new row to results table
            o.results_summary{end+1, :} = 0;
            o.results_summary.NumObs(end) = o.t;
            o.results_summary.Temperature(end) = o.g;
              
            % mutation ----------------------------------------------------
            % test the current mutation kernel
            fprintf('**testing %s mutation kernel with %d state particles **\n', upper(o.mutationKernel), o.Nx);
            [sjd1, msjd1, pen1] = TestCurrent(o);
            msjd = msjd1;
            
            % default values for alternative kernel
            sjd2 = zeros(1, o.m.np);
            
            % decide whether to test the alternative kernel
            test_alt_kernel = o.kernelAdaptation ~= "none";
            if o.kernelAdaptation == "lag"
                % always test the first 5 iterations
                test_alt_kernel = size(o.results_summary, 1) <= 5;
                
                if ~test_alt_kernel
                    ind_pmmh = find(o.results_summary.Pen_PMMH ~= 0, 1, 'last');
                    ind_pg = find(o.results_summary.Pen_PG ~= 0, 1, 'last');
                    
                    % calculate lag here
                    ind = min(ind_pmmh, ind_pg);
                    pen_pmmh = o.results_summary.Pen_PMMH(ind);
                    pen_pg = o.results_summary.Pen_PG(ind);
                    
                    if o.mutationKernel == "pg"
                        test_alt_kernel = ind_pg - ind_pmmh > pen_pmmh/pen_pg;
                    else
                        test_alt_kernel = ind_pmmh - ind_pg > pen_pg/pen_pmmh;
                    end
                end
            end
                
            % test the alternative kernel
            if test_alt_kernel
                % test the alternative mutation kernel
                SwitchKernel(o, true);
                fprintf('**testing %s mutation kernel with %d state particles **\n', upper(o.mutationKernel), o.Nx);
                [sjd2, msjd2, pen2] = TestCurrent(o);
            
                % if the fixed kernel has the better score
                if pen1 < pen2
                    SwitchKernel(o, true);
                else
                    msjd = msjd2;
                end 
            end

            % complete the mutation
            r = max(0, ceil((o.targetSJD - min(sjd1 + sjd2))./msjd*o.test_K)); %
            fprintf('**completing %d %s mutations **\n', r, upper(o.mutationKernel));
            for r = 1:r
                o.Mutation();
            end
            
            % switch mutation kernel back if necessary
            if o.fixed_kernel ~= o.mutationKernel
                SwitchKernel(o, true);
            end
            
            % update results ----------------------------------------------
            % calculate the SJD
            dtheta = dtheta - o.tsamples;
            sjd = mean((dtheta*o.tSig^(-0.5)).^2);
            
            o.results_summary.AR(end) = mean(dtheta ~= 0, 'all');
            o.results_summary.SJD(end) = min(sjd);
            
            o.target_means = [o.target_means; sum(o.tsamples .* exp(o.logNWt))];
            if o.results_summary.R_PMMH(end) > 0
                o.results_summary.AR_PMMH(end) = round(o.results_summary.AR_PMMH(end)/o.results_summary.R_PMMH(end), 3);
            end
            
            if o.results_summary.R_PG(end) > 0
                o.results_summary.AR_PG_1(end) = round(o.results_summary.AR_PG_1(end)/(o.results_summary.R_PG(end) * o.MALA_K), 3);
                o.results_summary.AR_PG_2(end) = round(o.results_summary.AR_PG_2(end)/(o.results_summary.R_PG(end) * o.MALA_K), 3);
            end
        end % adaptiveMutation
        
        
        %% Adaptation
        function [sjd, msjd, pen] = TestCurrent(o)
            % initialise
            dtheta = o.tsamples; 
            for j = 1:o.test_K
                o.Mutation(); 
            end

            % calculate SJD
            dtheta = dtheta - o.tsamples;
            sjd = real(mean((dtheta*o.tSig^(-0.5)).^2));
            msjd = max(min(sjd), 0.001);
            
            % calculate penalty
            pen = o.Nx / msjd;
            if o.mutationKernel == "pg"
                o.results_summary.Pen_PG(end) = pen;
            else
                o.results_summary.Pen_PMMH(end) = pen;
            end
        end % TestCurrent
        
        
        function SwitchKernel(o, switchk)
            % switch Nx
            if o.mutationKernel == "pg"
                o.Nx = o.Nx_PMMH;
            else
                o.Nx = o.Nx_PG;
            end
            
            % determine new kernel
            if switchk
                kernels = ["pg"; "pmmh"];
                o.mutationKernel = kernels(o.mutationKernel ~= kernels);
            end
            
            % switch to new kernel
            if o.mutationKernel == "pmmh"
                % update likelihood estimate
                [o.LL, o.xsamples] = o.loglikelihood(o.tsamples, o.Nx);
                o.Xk = [];
            else
                % update latent state trajectories
                drawLatentStates(o, o.Nx_PMMH);
                drawLatentStates(o, o.Nx);
                o.LL = [];
            end 
        end % SwitchKernel
        
    end % methods (Hidden)
    
    
  
end
