classdef StochasticVolatilityInMean_model < handle & abstractSSM 
    
    properties
        % constants
        A = 1; B = 2; D = 3; LS = 4; PHI = 5; LSIGMA = 6;
        y0;
    end
    
    methods
        %% Constructor
        function o = StochasticVolatilityInMean_model(T, y0, varargin)
            
            % parse optional input
            p = inputParser;
            addParameter(p, 'theta', nan, @(x) length(x) == 6 || isnan(x));
            addParameter(p, 'y', nan); 
            addParameter(p, 'x', nan);
            parse(p, varargin{:});
            
            % set values from input
            o.T = T;
            o.y0 = y0;
            o.theta = p.Results.theta;
            o.x = p.Results.x;
            o.y = p.Results.y;
            o.parameter_blocks = {[o.A, o.B, o.D, o.LS], [o.PHI, o.LSIGMA]};
            
            % simulate data if necessary
            if isnan(p.Results.y) 
                if any(isnan(o.theta))
                    o.theta = [0 1 0 2 0.5 2]; 
                end 
                o.x = o.simulate_state_trajectory(o.theta);
                o.y = y_rnd(o, o.x, o.theta, o.nty); 
            end
            
            % fixed values
            o.np = 6;
            o.tnames = {'A'; 'B*'; 'D'; 'LOG(S)'; 'PHI*'; 'LOG(SIGMA)'};
            o.names = {'A'; 'B'; 'D'; 'S'; 'PHI'; 'SIGMA'};
        end % StochasticVolatility_model(T, y0, varargin)
        
        %% Observed process
        
        % calculate the log-pdf of the observation density
        function lp = y_lpdf(o, t, x, theta)
            a = theta(:, o.A);
            b = inverse_logit(o, theta(:, o.B));
            d = theta(:, o.D);
            
            if t == 1
                mu = a + b*o.y0 + d.*exp(2*theta(:, o.LS) + x);
            else
                mu = a + b*o.y(t-1) + d.*exp(2*theta(:, o.LS) + x);
            end
            sig = exp(theta(:, o.LS) + 0.5*x);
            lp = norm_lpdf(o.y(t), mu, sig);
            lp(isnan(lp)) = -inf;
        end % y_lpdf(o, t, x, theta)
        
        
        % simulate from the observation density
        function y = y_rnd(o, x, theta)
            a = theta(:, o.A);
            b = inverse_logit(o, theta(:, o.B));
            d = theta(:, o.D);
            
            y = zeros(size(x));
            y = [o.y0 ; y]; % add y0
            for i = 1:length(y)-1
                mu = a + b*y(i) + d.*exp(2*theta(:, o.LS) + x(i));
                sig = exp(theta(:, o.LS) + 0.5*x(i));
                y(i+1) = mu + sig.*randn(); 
            end
            y = y(2:end); % remove y0
        end % y_rnd(o, x, theta)
        
        
        %% Latent process
        
        % calculate log-pdf of x1
        function lp = x1_lpdf(o, theta, x1)
            phi = inverse_logit(o, theta(:, o.PHI)); 
            lp = norm_lpdf(x1, 0, exp(theta(:, o.LSIGMA) - 0.5*log(1-phi.^2)));
        end % x1_lpdf(o, theta, x1)
        
        
        % calculate the log-pdf of the transition density
        function lp = x_lpdf(o, x_current, x_prev, theta)
            phi = inverse_logit(o, theta(:, o.PHI));
            sig = exp(theta(:, o.LSIGMA));
            
            lp = norm_lpdf(x_current, phi.*x_prev, sig);
            lp(isnan(lp)) = -inf;
        end % x_lpdf(o, x_current, x_prev, theta)
        
        
        % simulate the latent state at t = 1
        function x1 = x1_rnd(o, theta, varargin)
            if nargin == 3; N = varargin{:};
            else; N = 1; end

            phi = inverse_logit(o, theta(:, o.PHI)); 
            x1 = exp(theta(:, o.LSIGMA) - 0.5*log(1-phi.^2)).*randn(size(theta, 1), N);
        end % x1_rnd(o, theta, varargin)
        
        
        % simulate from the transition density
        function x = x_rnd(o, xa, theta, ~, ~, varargin)
            phi = inverse_logit(o, theta(:, o.PHI));    
            sig = exp(theta(:, o.LSIGMA));
            x = phi.*xa + sig.*randn(size(xa));
        end % x_rnd(o, xa, theta, ~, ~, varargin)
        
        
        %% Prior

        % calculate the log-pdf of the prior
        function lp = prior_lpdf(o, theta)
            lp = norm_lpdf(theta(:, o.A), 0, 10)...
                + sum(theta(:, o.B) - 2*log(1 + exp(theta(:, o.B))), 2)...
                + norm_lpdf(theta(:, o.D), 0, 10)...
                + logHN_lpdf(theta(:, o.LS), 2)...
                + sum(theta(:, o.PHI) - 2*log(1 + exp(theta(:, o.PHI))), 2)... 
                + logHN_lpdf(theta(:, o.LSIGMA), 2);
        end % prior_lpdf(o, theta)
        
        
        % simulate from the prior
        function draws = prior_rnd(o, N)
            draws = zeros(N, o.np);
            draws(:, o.A) = normrnd(0, 10, N, 1);
            draws(:, o.B) = unifrnd(0, 1, N, 1); 
            draws(:, o.D) = normrnd(0, 10, N, 1);
            draws(:, o.LS) = log(abs(normrnd(0, 2, N, 1)));
            draws(:, o.PHI) = unifrnd(0, 1, N, 1); 
            draws(:, o.LSIGMA) = log(abs(normrnd(0, 2, N, 1)));
            
            draws(:, [o.B o.PHI]) = log(draws(:, [o.B o.PHI])) - log((1 - draws(:, [o.B o.PHI])));
        end % prior_rnd(o, N)
        
        
        %% Other
        
        % inverse logit transformation
        function p = inverse_logit(~, p)
            p = exp(p - log(1 + exp(p)));
        end
        
        
        % transform the parameters, can be the inverse transformation
        function theta = transform(o, theta, inverse)
            if inverse
                theta(:, [o.LS o.LSIGMA]) = exp(theta(:, [o.LS o.LSIGMA]));
                theta(:, [o.B o.PHI]) = exp(theta(:, [o.B o.PHI]))./(1 + exp(theta(:, [o.B o.PHI])));
            else 
                theta(:, [o.LS o.LSIGMA]) = log(theta(:, [o.LS o.LSIGMA]));
                theta(:, [o.B o.PHI]) = log(theta(:, [o.B o.PHI])./(1 - theta(:, [o.B o.PHI])));
            end
        end
        
        
        % calculate the gradients of the log-posterior
        function f = grad_post(o, t, x, theta, G)
            y = o.y(1:t);
            [m, n] = size(theta);
            f = zeros(m, n);
            
            % parameters 
            a = theta(:, o.A);
            b = inverse_logit(o, theta(:, o.B));
            d = theta(:, o.D);
            s_sq = exp(2*theta(:, o.LS));
            phi = inverse_logit(o, theta(:, o.PHI)); 
            sig_sq = exp(2*theta(:, o.LSIGMA));
            sig_isq = exp(-2*theta(:, o.LSIGMA));
            
            % common terms
            C = y' - a - b.*[o.y0 y(1:end-1)'] - d.*exp(2*theta(:, o.LS) + x);
            H = x(:, 2:end) - phi.*x(:, 1:end-1);
            ddphi = phi - phi.^2; 
            
            % gradients
            f(:, o.A) = G*sum(exp(-2*theta(:, o.LS) - x).*C, 2) - a/10^2;
            f(:, o.B) = 1 - 2*b + G.*(b - b.^2).*sum(exp(-2*theta(:, o.LS) - x).*C.*[o.y0 y(1:end-1)'], 2);
            f(:, o.D) = G*sum(C, 2) - d/10^2;
            f(:, o.LS) = 1 - t*G - s_sq/2^2 + 2*G*d.*sum(C, 2) + G*sum(exp(-2*theta(:, o.LS) - x).*C.^2, 2);
            
            f(:, o.PHI) = 1 - 2*phi + exp(2*log(phi) - log(1 + phi)) + sig_isq.*ddphi.*sum(H.*x(:, 1:end-1), 2) + x(:, 1).^2.*exp(-2*theta(:, o.LSIGMA) + 2*log(phi) + log(1-phi));
            f(:, o.LSIGMA) = -sig_sq/2^2 + 1 - t + sig_isq .* sum(H.^2, 2) + sig_isq.*(1-phi.^2).*x(:, 1).^2;
        end % grad_post(o, y, x, theta, G)
        
    end % methods
    
end % StochasticVolatility_model

%% Helper functions
% log pdf of the normal distribution
function y = norm_lpdf(x, m, s)
    y = -0.5*log(2*pi) - log(s) - 0.5*(x - m).^2./exp(2*log(s)); 
end

% log pdf of the log half-normal distribution
function y = logHN_lpdf(logx, s)
    y = 0.5*log(2/pi)-log(s)-exp(2*logx)/(2*s^2) + logx;
end