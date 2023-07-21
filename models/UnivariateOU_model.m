classdef UnivariateOU_model < handle & abstractSSM 
    properties
        % constants
        LPHI = 1; MU = 2; LSIGMA = 3; BETA = 4; 
        z;
        dbeta;
    end
    
    methods
        %% constructor
        function o = UnivariateOU_model(T, dbeta, varargin)
            
            % parse optional input
            p = inputParser;
            addParameter(p, 'theta', nan, @(x) length(x) == 3 + dbeta || isnan(x));
            addParameter(p, 'y', nan); 
            addParameter(p, 'x', nan);
            addParameter(p, 'z', nan);
            parse(p, varargin{:});
            
            % set values from input
            o.T = T;
            o.theta = p.Results.theta;
            o.x = p.Results.x;
            o.y = p.Results.y;
            o.z = p.Results.z;
            o.dbeta = dbeta;
            o.parameter_blocks = {1:3, 4:(dbeta+3)};
            
            % simulate data if necessary
            if isnan(p.Results.y) 
                if any(isnan(o.theta))
                    error("Need to specify a value for theta");
                end 
                o.x = o.simulate_state_trajectory(o.theta);
                [o.y, o.z] = y_rnd(o, o.x, o.theta);                 
            end
                        
            % fixed values
            o.np = 3 + dbeta;
            o.tnames = cell(o.np, 1);
            o.tnames(1:3) = {'LOG(PHI)';'MU';'LOG(SIGMA)'};
            o.tnames(4:end) = num2cell("BETA" + (1:o.dbeta)'); 
            
            o.names = o.tnames;
            o.names([1 3]) = {'PHI'; 'SIGMA'}; 
        end % FlexibleAllee_model(T, varargin)
        
        
        %% Observed process
        
        % calculate the log-pdf of the observation density
        function lp = y_lpdf(o, t, xt, theta)  
            mu = theta(:, o.BETA:end)*o.z(t, :)'; %Nt x 1   sum(theta(:, o.BETA:end).*o.z(t, :), 2)
            sig = exp(0.5*xt); % Nt x Nx
            lp = norm_lpdf(o.y(t), mu, sig); % Nt x Nx
        end % y_lpdf(o, t, x, theta) 
        
        
        % simulate from the observation density
        % combine this with simulation for z
        function [y, z] = y_rnd(o, xt, theta) 
            z = normrnd(0, 1, o.T, o.dbeta);
            y = z*theta(:, o.BETA:end)' + exp(0.5*xt).*randn(size(xt));
        end % y_rnd(o, logx, theta) 
        
        
        %% Latent process
        
        % calculate log-pdf of x1
        function lp = x1_lpdf(o, theta, x1)
            theta = o.transform(theta, true);
            
            sig = theta(:, o.LSIGMA)./(1-theta(:, o.LPHI).^2);
            lp = norm_lpdf(x1, theta(:, o.MU), sig); % Nt x 1
        end % x1_lpdf(m, theta, x1)
        
        
        % calculate the log-pdf of the transition density
        function lp = x_lpdf(o, x_new, x, theta)
            theta = o.transform(theta, true);
            
            mu = theta(:, o.MU) + theta(:, o.LPHI).*(x - theta(:, o.MU));
            sig = theta(:, o.LSIGMA);
            lp = norm_lpdf(x_new, mu, sig); % Nt x 1
        end % x_lpdf(o, logx_current, logx_prev, theta)
        
        
        % simulate the latent state at t = 1
        function x1 = x1_rnd(o, theta, varargin)
            if nargin == 3; N = varargin{:};
            else; N = 1; end
            
            theta = o.transform(theta, true);
            
            sig = theta(:, o.LSIGMA)./sqrt(1-theta(:, o.LPHI).^2);
            x1 = theta(:, o.MU) + sig.*randn(size(theta, 1), N); % Nt X Nx
        end % x1_rnd(o, theta, varargin)
        
        
        % simulate from the transition density
        function x_new = x_rnd(o, x, theta, ~, ~, varargin)
            theta = o.transform(theta, true);
            
            mu = theta(:, o.MU) + theta(:, o.LPHI).*(x - theta(:, o.MU));
            sig = theta(:, o.LSIGMA);
            x_new = mu + sig.*randn(size(x)); % Nt x Nx
        end 
        
        
        %% Prior
        
        % calculate the log-pdf of the prior
        function lp = prior_lpdf(o, theta)
            lp = theta(:, o.LPHI) - 2*log(1 + exp(theta(:, o.LPHI))) ... %logHN_lpdf(theta(:, o.LPHI), 10) ...
                + norm_lpdf(theta(:, o.MU), 0, 5) ...
                + logHN_lpdf(theta(:, o.LSIGMA), 10) ...
                + sum(norm_lpdf(theta(:, o.BETA:end), 0, 1), 2);
        end % prior_lpdf(o, theta)
        
        
        % simulate from the prior
        function draws = prior_rnd(o, N)
            draws = zeros(N, o.np);
            draws(:, o.LPHI) = rand(N, 1); %log(abs(normrnd(0, 10, N, 1)));
            draws(:, o.LPHI) = log(draws(:, o.LPHI)./(1-draws(:, o.LPHI)));
            draws(:, o.MU) = normrnd(0, 5, N, 1);
            draws(:, o.LSIGMA) = log(abs(normrnd(0, 10, N, 1)));
            draws(:, o.BETA:end) = normrnd(0, 1, N, o.dbeta);
        end % prior_rnd(o, N)
        
        
        %% Other
        
        % transform the parameters, can be the inverse transformation
        function theta = transform(o, theta, inverse)
            if inverse
                theta(:, o.LSIGMA) = exp(theta(:, o.LSIGMA));
                theta(:, o.LPHI) = exp(theta(:, o.LPHI) - log(1 + exp(theta(:, o.LPHI))));
            else
                theta(:, o.LSIGMA) = log(theta(:, o.LSIGMA));
                theta(:, o.LPHI) = log(theta(:, o.LPHI)) - log(1 - theta(:, o.LPHI));
            end
        end % transform(o, theta, inverse)
        
        
        % calculate the gradients of the log-posterior
        function f = grad_post(o, t, x, theta, G)
            f=zeros(size(theta));
            
            % common terms
            d = exp(theta(:, o.LPHI));
            a = 1 + 2*d;
            b = 1 + d;
            c = exp(-2*theta(:, o.LSIGMA));
            
            if t == 1
                summand_x = 0;
            else
                summand_x = x(:, 2:end) - theta(:, o.MU) - d./b.*(x(:, 1:end-1)-theta(:, o.MU));
            end
            
            f(:, o.LPHI) = d./a + 1 - 3*d./b + c.*d.^2./(2*b.^3).*(x(:, 1) - theta(:, o.MU)).^2 ...
                - d.*a.*c./b.^2.*sum(summand_x.*(x(:, 1:end-1)-theta(:, o.MU)), 2);
            
            f(:, o.MU) = -theta(:, o.MU)/5^2 + c.*a./b.^2.*(x(:, 1) - theta(:, o.MU)) ...
                + c./b.*sum(summand_x, 2);
            
            f(:, o.LSIGMA) = -(t-1) - 1./(10^2*c) + c.*a./b.^2.*(x(:, 1) - theta(:, o.MU)).^2 ...
                + c.*sum(summand_x.^2, 2);
            
            for i = o.BETA:(o.BETA + o.dbeta - 1)
                f(:, i) = -theta(:, i) + G.*sum(exp(-x).*o.z(1:t, i-o.BETA+1)'.*(o.y(1:t)' - theta(:, o.BETA:end)*o.z(1:t, :)'), 2);
            end
            
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

