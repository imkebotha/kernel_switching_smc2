classdef FlexibleAllee_model < handle & abstractSSM 

    properties
        % constants
        LX0 = 1; BETA0 = 2; BETA1 = 3; BETA2 = 4; 
        LGAMMA = 5; LSIGMA = 6; 
        a; % prior standard deviations of the beta parameters
    end
    
    methods
        %% constructor
        function o = FlexibleAllee_model(T, varargin)
            
            % parse optional input
            p = inputParser;
            addParameter(p, 'theta', nan, @(x) length(x) == 6 || isnan(x));
            addParameter(p, 'y', nan); 
            addParameter(p, 'x', nan);
            parse(p, varargin{:});
            
            % set values from input
            o.T = T;
            o.theta = p.Results.theta;
            o.x = p.Results.x;
            o.y = p.Results.y;
            o.parameter_blocks = {1:5, 6};
            
            % simulate data if necessary
            if isnan(p.Results.y) 
                if any(isnan(o.theta))
                    o.theta = [6.3 0 0 0 -2.3 -4.6];
                end 
                o.x = o.simulate_state_trajectory(o.theta);
                o.y = y_rnd(o, o.x, o.theta);                 
            end
                        
            % fixed values
            o.a = [0.2 0.001 0.001];
            o.np = 6;
            o.tnames = {'LOG(X0)';'BETA0';'BETA1';'BETA2';'LOG(GAMMA)';'LOG(SIGMA)'};
            o.names = {'LOG(X0)';'BETA0';'BETA1';'BETA2';'GAMMA';'SIGMA'};
        end % FlexibleAllee_model(T, varargin)
        
        
        %% Observed process
        
        % calculate the log-pdf of the observation density
        function lp = y_lpdf(o, t, x, theta)  
            sig = exp(theta(:, o.LSIGMA));
            lp = norm_lpdf(o.y(t), x, sig);
            lp(isnan(lp)) = -inf;
        end % y_lpdf(o, t, x, theta) 
        
        
        % simulate from the observation density
        function y = y_rnd(o, logx, theta) 
            sig = exp(theta(:, o.LSIGMA));
            y = logx + sig*randn(size(logx));
        end % y_rnd(o, logx, theta) 
        
        
        %% Latent process
        
        % calculate log-pdf of x1
        function lp = x1_lpdf(m, theta, x1)
            lp = m.x_lpdf(x1, theta(:, m.LX0), theta);
        end % x1_lpdf(m, theta, x1)
        
        
        % calculate the log-pdf of the transition density
        function logf = x_lpdf(o, logx_current, logx_prev, theta)
           b0 = theta(:, o.BETA0);
           b1 = theta(:, o.BETA1);
           b2 = theta(:, o.BETA2);
           
           mu = logx_prev + b0  + b1.*exp(logx_prev) + b2.*exp(2*logx_prev);
           
           gam = exp(theta(:, o.LGAMMA));
           logf = norm_lpdf(logx_current, mu, gam);
           logf(isnan(logf)) = -inf;
        end % x_lpdf(o, logx_current, logx_prev, theta)
        
        
        % simulate the latent state at t = 1
        function logx1 = x1_rnd(o, theta, varargin)
            logx0 = theta(:, o.LX0); 
            if nargin == 3
                N = varargin{:};
                logx0 = repmat(logx0, 1, N);
            end
            logx1 = o.x_rnd(logx0, theta, 0, 1); 
        end % x1_rnd(o, theta, varargin)
        
        
        % simulate from the transition density
        function logx = x_rnd(o, logxa, theta, ~, ~, varargin)
            b0 = theta(:, o.BETA0);
            b1 = theta(:, o.BETA1);
            b2 = theta(:, o.BETA2);
            g = exp(theta(:, o.LGAMMA));
            
            logx = logxa + b0 + b1.*exp(logxa) + b2.*exp(2*logxa) + g.*randn(size(logxa)); 
        end % x_rnd(o, logxa, theta, ~, ~, varargin)
        
        
        %% Prior
        
        % calculate the log-pdf of the prior
        function lp = prior_lpdf(o, theta)
            lp = logHN_lpdf(theta(:, o.LX0), 1000) ...
                + norm_lpdf(theta(:, o.BETA0), 0, o.a(1)) ...
                + norm_lpdf(theta(:, o.BETA1), 0, o.a(2)) ...
                + norm_lpdf(theta(:, o.BETA2), 0, o.a(3)) ...
                + logExp_lpdf(theta(:, o.LGAMMA), 1) ...
                + logExp_lpdf(theta(:, o.LSIGMA), 1);
        end % prior_lpdf(o, theta)
        
        
        % simulate from the prior
        function draws = prior_rnd(o, N)
            draws = zeros(N, o.np);
            draws(:, o.LX0) = log(abs(normrnd(0, 1000, N, 1)));
            draws(:, o.BETA0) = normrnd(0, o.a(1), N, 1);
            draws(:, o.BETA1) = normrnd(0, o.a(2), N, 1);
            draws(:, o.BETA2) = normrnd(0, o.a(3), N, 1);
            draws(:, [o.LGAMMA o.LSIGMA]) = log(exprnd(1, N, 2));
        end % prior_rnd(o, N)
        
        
        %% Other
        
        % transform the parameters, can be the inverse transformation
        function theta = transform(o, theta, inverse)
            params = [o.LGAMMA o.LSIGMA];
            if inverse; theta(:, params) = exp(theta(:, params));
            else; theta(:, params) = log(theta(:, params));
            end
        end % transform(o, theta, inverse)
        
        
        % calculate the gradients of the log-posterior
        function f = grad_post(o, t, logx, theta, G)
            logy = o.y(1:t);
            [m, n] = size(theta);
            f = zeros(m, n);
            
            % parameters
            lx0 = theta(:, o.LX0);
            b0 = theta(:, o.BETA0);
            b1 = theta(:, o.BETA1);
            b2 = theta(:, o.BETA2);
            lgamma = theta(:, o.LGAMMA);
            lsigma = theta(:, o.LSIGMA);
            
            % common terms
            logxf = [lx0 logx];
            xf = exp(logxf(:,1:end-1));
            xf_sq = exp(2*logxf(:,1:end-1));
            C = logx - logxf(:, 1:end-1) - b0 - b1.*xf - b2.*xf_sq;
            C_sq = C.^2;
            
            D_sq = sum((logy' - logx).^2, 2); 
            g_isq = exp(-2*lgamma);
            
            % gradients
            f(:, o.LX0) = g_isq.*C(:, 1).*(1 + b1.*exp(lx0) + 2.*b2.*exp(2*lx0)) - exp(2*lx0)/1000^2 + 1;
            f(:, o.BETA0) = -b0/o.a(1)^2 + g_isq.*sum(C, 2);
            f(:, o.BETA1) = -b1/o.a(2)^2 + g_isq.*sum(C.*xf, 2);
            f(:, o.BETA2) = -b2/o.a(3)^2 + g_isq.*sum(C.*xf_sq, 2);
            f(:, o.LGAMMA) = -exp(lgamma) + 1 - t + g_isq.*sum(C_sq, 2);
            f(:, o.LSIGMA) = -exp(lsigma) + 1 - t*G + G*exp(-2*lsigma).*D_sq;
        end % grad_post(o, logy, logx, theta, G)

    end % methods
    
end % FlexibleAllee_model

%% Helper functions
% log pdf of the normal distribution
function y = norm_lpdf(x, m, s)
    y = -0.5*log(2*pi) - log(s) - 0.5*(x - m).^2./exp(2*log(s)); 
end

% log pdf of the log-transformed half-normal distribution
function y = logHN_lpdf(logx, s)
    y = 0.5*log(2/pi)-log(s)-exp(2*logx)/(2*s^2) + logx;
end

% log pdf of the log-transformed exponential distribution
function y = logExp_lpdf(logx, g)
    y = log(g) - g*exp(logx) + logx;
end
