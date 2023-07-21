classdef abstractSSM < handle
    
    % time discretisation is assumed to be 1
    properties
        y;          % observations
        x;          % latent states (true) if known, initial values otherwise
        theta;      % parameters (true) if known, initial values otherwise
        
        T;          % length of time series
        np;         % number of parameters
        tnames;     % names of transformed parameters
        names;      % names of untransformed parameters
        
        parameter_blocks;   % update blocks for the parameters
    end
    
    %% Common method/s
    methods 

        % calculate the log of the observation density over all the
        % observations
        function logf = log_obs_density(m, ny, theta, x)
            logf = zeros(size(theta, 1), 1);
            for i = 1:ny
                logf = logf + m.y_lpdf(i, x(:, i), theta);
            end
        end
        
        % simulate a full state trajectory from the transition density
        function x = simulate_state_trajectory(o, theta)
            x = zeros(o.T, 1);
            x(1) = x1_rnd(o, theta, 1); 
            for i = 2:o.T
                x(i) = x_rnd(o, x(i-1), theta); 
            end
        end
        
        % calculate the log-density over all the latent states
        function lp = log_state_density(m, theta, Xk)
            [Nt, t] = size(Xk);
            lp = zeros(Nt, t);
            lp(:, 1) = x1_lpdf(m, theta, Xk(:, 1)); %m.x_lpdf(Xk(:, 1), theta(:, m.X0), theta);
            if t > 1
                lp(:, 2:end) = m.x_lpdf(Xk(:, 2:end), Xk(:, 1:end-1), theta);
            end
        end
    end
  
    %% Default methods
    methods
        % variable transformation
        % default: no transformation
        function new_theta = transform(~, theta, varargin)
            new_theta = theta;
        end
        
        % log of the density for the initial state P(x_{1}|theta)
        % default: x_{1} is treated as one of the unknown parameters and
        % estimated
        function lp = x1_lpdf(varargin)
            lp = 0;
        end
        
        % log of the transition density P(x_{t}|x_{t-1}, theta)
        % default: function is intractable. This method will only be called
        % to update parameters whose full conditional distributions don't
        % require the transition density. In this case, the transition
        % density will be a constant
        function lp = x_lpdf(varargin)
           lp = 0;
        end
    end 
    
    %% Abstract methods
    % These methods must be implemented, but are model specific 
    methods (Abstract)
        % obervation density
        y_lpdf(o);
        y_rnd(o);
        
        % transition density
        x1_rnd(o);
        x_rnd(o);
        
        % prior
        prior_lpdf(o);
        prior_rnd(o);
    end
end