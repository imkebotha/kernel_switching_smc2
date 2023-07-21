classdef BrownianMotion_model < handle & abstractSSM 
      
    properties
        % constants
        X0 = 1; BETA = 2; LGAMMA = 3; LSIGMA = 4;
    end
    
    methods
        %% Constructor
        function o = BrownianMotion_model(T, varargin)
            
            % parse optional input
            p = inputParser;
            addParameter(p, 'theta', nan, @(x) length(x) == 4 || isnan(x));
            addParameter(p, 'y', nan); 
            addParameter(p, 'x', nan);
            parse(p, varargin{:});
            
            % set values from input
            o.theta = p.Results.theta;
            o.x = p.Results.x;
            o.y = p.Results.y;
            o.T = T;
            o.parameter_blocks = {1:3, 4};
            
            % simulate data if necessary
            if isnan(p.Results.y) 
                if any(isnan(o.theta))
                    o.theta = [3.5 2 0 -0.7]; 
                end 
                o.x = o.simulate_state_trajectory(o.theta);
                o.y = y_rnd(o, o.x, o.theta); 
            end
            
            % fixed values
            o.np = 4;
            o.tnames = {'X_0';'BETA';'LOG(GAMMA)';'LOG(SIGMA)'};
            o.names = {'X_0';'BETA';'GAMMA';'SIGMA'};
        end % logGBM_model(T, varargin)
        
        
        %% Observed process
        
        % calculate the log-pdf of the observation density
        function logf = y_lpdf(o, t, xt, theta)  
            sig = exp(theta(:, o.LSIGMA));
            logf = norm_lpdf(o.y(t), xt, sig);% Nt x Nx
        end
        
        
        % simulate from the observation density
        function f = y_rnd(o, x, theta)  
            f = x + exp(theta(:, o.LSIGMA)).*randn(size(x));
        end
        
        
        %% Latent process
        
        % calculate log-pdf of x1
        function lp = x1_lpdf(m, theta, x1)
            lp = m.x_lpdf(x1, theta(:, m.X0), theta); % Nt x 1
        end % x1_lpdf(m, theta, x1)
        
        
        % calculate the log-pdf of the transition density
        function logf = x_lpdf(o, x_current, x_prev, theta)
           mu = x_prev + theta(:, o.BETA) - 0.5*exp(2*theta(:, o.LGAMMA));
           gam = exp(theta(:, o.LGAMMA));
           logf =  norm_lpdf(x_current, mu, gam); % Nt x 1
        end % x_lpdf(o, x_current, x_prev, theta)
        
        
        % simulate the latent state at t = 1
        function x1 = x1_rnd(o, theta, varargin)
            x0 = theta(:, o.X0); 
            if nargin == 3
                N = varargin{:};
                x0 = repmat(x0, 1, N);
            end
            x1 = o.x_rnd(x0, theta, 0, 1);  % Nt x Nx
        end % x1_rnd(o, theta, varargin)
        
        
        % simulate from the transition density
        function x = x_rnd(o, xa, theta, ~, ~, varargin)
            b = theta(:, o.BETA);
            g = exp(theta(:, o.LGAMMA));
            gsq = exp(2*theta(:, o.LGAMMA));
            
            x = xa + b - 0.5*gsq + g.*randn(size(xa)); % Nt x Nx
        end % x_rnd(o, xa, theta, ~, ~, varargin)

        
        %% Prior
        
        % calculate the log-pdf of the prior
        function lp = prior_lpdf(o, theta)
            lp = norm_lpdf(theta(:, o.X0), 3, 5)...
                + norm_lpdf(theta(:, o.BETA), 2, 5)...
                + logHN_lpdf(theta(:, o.LGAMMA), 2)...
                + logHN_lpdf(theta(:, o.LSIGMA), 2);
        end
        
        
        % simulate from the prior
        function draws = prior_rnd(o, N)
            mvn_mu = [3 2 0 0];
            mvn_cov = diag([5 5 2 2].^2);

            draws = mvnrnd(mvn_mu, mvn_cov, N);
            draws(:, [o.LGAMMA o.LSIGMA]) = log(abs(draws(:, [o.LGAMMA o.LSIGMA])));
        end
        
        
        %% Other
        
        % transform the parameters, can be the inverse transformation
        function theta = transform(o, theta, inverse)
            params = [o.LGAMMA o.LSIGMA];
            if inverse; theta(:, params) = exp(theta(:, params));
            else; theta(:, params) = log(theta(:, params));
            end
        end % transform(o, theta, inverse)
        
        
        % calculate the gradients of the log-posterior
        function f = grad_post(o, t, x, theta, G)
            y = o.y(1:t);
            [m, n] = size(theta);
            f = zeros(m, n);
            
            % parameters
            lgamma = theta(:, o.LGAMMA);
            lsigma = theta(:, o.LSIGMA);
            x0 = theta(:, o.X0);
            b = theta(:, o.BETA);
            
            % common terms
            g_sq = exp(2*lgamma);
            g_isq = exp(-2*lgamma);
            C = x - [x0 x(:, 1:end-1)] - b + 0.5*g_sq;
            C_sq = C.^2;
            
            % gradients
            f(:, o.X0) = -(x0 - 3)/25 + g_isq.*C(:, 1);
            f(:, o.BETA) = - (b - 2)/25 + g_isq.*sum(C, 2);
            f(:, o.LGAMMA) = - t - sum(C, 2) + g_isq.*sum(C_sq, 2) - g_sq/4 + 1;
            f(:, o.LSIGMA) = - G*t + G*exp(-2*lsigma).*sum((y' - x).^2, 2) - exp(2*lsigma)/4 + 1;
        end % grad_post(o, y, x, theta, G)
        
    end % methods

end

%% Helper functions
% log pdf of the normal distribution
function y = norm_lpdf(x, m, s)
    y = -0.5*log(2*pi) - log(s) - 0.5*(x - m).^2./exp(2*log(s)); 
end

% log pdf of the log half-normal distribution
function y = logHN_lpdf(logx, s)
    y = 0.5*log(2/pi)-log(s)-exp(2*logx)/(2*s^2) + logx;
end

