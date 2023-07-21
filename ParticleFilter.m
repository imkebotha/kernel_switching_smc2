classdef ParticleFilter

    methods(Static)
        %% single iteration of the standard particle filter
        function [LLt, Xt] = iteration(m, theta, Nx, t, Xt)
            if t == 1
                Xt = m.x1_rnd(theta, Nx);
                logNWt = - log(Nx);
            elseif t > 1
                logWt = m.y_lpdf(t-1, Xt, theta) - log(Nx); 
                logNWt = logWt - logsumexp(logWt, 2);
                logNWt(isnan(logNWt)) = -inf;
                
                % resample
                for p = 1:size(logNWt, 1) 
                    if ~all(isinf(logNWt(p, :)))
                        I = randsample(1:Nx, Nx, true, exp(logNWt(p, :)));
                        Xt(p, :, :) = Xt(p, I, :);
                        logNWt(p, :) = -log(Nx);
                    end
                end
                
                % simulate model
                Xt = m.x_rnd(Xt, theta, t-1, t);
            end

            % update weights
            logWt = m.y_lpdf(t, Xt, theta) + logNWt; %m.y(t, :)

            % incremental log-likelihood estimate
            LLt = logsumexp(logWt, 2);
            LLt(isnan(LLt)) = -inf;
        end % iteration
        
        %% standard bootstrap particle filter with adaptive resampling
        function [LL, Xt, logWt] = standard(m, theta, Nx, varargin)
            % parse optional input
            p = inputParser;
            addParameter(p, 'T', m.T);
            addParameter(p, 'g', 1);
            parse(p, varargin{:});
            
            T = p.Results.T;
            g = p.Results.g;

            % at t = 1
            Xt = m.x1_rnd(theta, Nx); 
            logWt = g*m.y_lpdf(1, Xt, theta) - log(Nx); 
            logWt(isnan(logWt)) = -inf;
            if all(logWt == -inf)
                LL = -inf;
                return
            end
   
            logNWt = logWt - logsumexp(logWt, 2);
            LL = logsumexp(logWt, 2); 
            for t = 2:T 
                % adaptive resampling
                deficientESS = exp(-logsumexp(2*logNWt, 2)) < Nx/2;
                if (any(deficientESS))
                    for p = find(deficientESS') 
                        I = randsample(1:Nx, Nx, true, exp(logNWt(p, :)));
                        Xt(p, :, :) = Xt(p, I, :);
                        logNWt(p, :) = -log(Nx);
                    end
                end

                % simulate model 
                Xt = m.x_rnd(Xt, theta, t-1, t); 
                
                % update weights
                logWt = g*m.y_lpdf(t, Xt, theta) + logNWt; %m.y(t, :)
                logNWt = logWt - logsumexp(logWt, 2);
                
                % update likelihood estimate
                ll_increment = logsumexp(logWt, 2);
                LL = LL + ll_increment;
            end 
            LL(isnan(LL)) = -inf;
        end % standard
        

        %% conditional particle filter
        function [LL, Xm, logNm] = conditional(m, theta, Nx, Xk, varargin)

            % parse optional input
            p = inputParser;
            addParameter(p, 'g', 1);
            addParameter(p, 'T', m.T);
            parse(p, varargin{:});
            
            g = p.Results.g;
            T = p.Results.T;
            Nt = size(theta, 1);

            % initialise
            Xm = zeros(Nt, Nx, T);
            logNm = zeros(Nt, Nx, T);
            Xt = zeros(Nt, Nx);
            
            % always store invariant path/s at the top
            havepath = true;
            if isempty(Xk)
                havepath = false;
                notkx = 1:Nx;
                N = Nx;
            else
                kx = 1; notkx = 2:Nx; 
                N = Nx - 1;
            end
            
            Xt(:, notkx, :) = m.x1_rnd(theta, N);
            
            if havepath; Xt(:, kx, :) = Xk(:, 1, :); end
            Xm(:, :, 1) = Xt;

            logWt = g*(m.y_lpdf(1, Xt, theta)) - log(Nx); % m.y(1, :)
            logNWt = logWt - logsumexp(logWt, 2);
            LL = logsumexp(logWt, 2);
            logNm(:, :, 1) = logWt; 
            for t = 2:T
                
                % adaptive resampling
                deficientESS = exp(-logsumexp(2*logNWt, 2)) < Nx/2;
                if (any(deficientESS))
                    for p = find(deficientESS')
                        if ~any(isnan(logNWt(p, :))) 
                            I = randsample(1:Nx, N, true, exp(logNWt(p, :)));
                            Xt(p, notkx, :) = Xt(p, I, :);
                            logNWt(p, :) = -log(Nx);
                        end
                    end
                end
                
                % simulate model 
                if havepath; Xt(:, kx, :) = Xk(:, t, :); end
                Xt(:, notkx, :) = m.x_rnd(Xt(:, notkx, :), theta, t-1, t);

                % update weights
                logWt = g*(m.y_lpdf(t, Xt, theta)) + logNWt; 
                logNWt = logWt - logsumexp(logWt, 2);

                % update likelihood estimate
                LL = LL + logsumexp(logWt, 2);

                Xm(:, :, t) = Xt;
                logNm(:, :, t) = logWt; 
            end
    
            LL(isnan(LL)) = -inf;
            logNm(isnan(logNm)) = -inf;
        end

        %% backward sampling
        
        function Xk = drawB(m, Xm, logWx, theta)
            [Nt, Nx, T] = size(Xm);
            Xk = zeros(Nt, T);
            
            logNWx = logWx(:, :, end) - logsumexp(logWx(:, :, end), 2);
            logNWx(isnan(logNWx)) = -inf;

            % draw final values
            for i = 1:Nt
                if all(isinf(logNWx(i, :)))
                    k = randsample(1:Nx, 1, true);
                else
                    k = randsample(1:Nx, 1, true, exp(logNWx(i, :)));
                end
                
                Xk(i, T, :) = Xm(i, k, T);
            end
            
            for t = T-1:-1:1 
                logwx = logWx(:, :, t) + m.x_lpdf(Xk(:, t + 1, :), Xm(:, :, t, :), theta);
                lognwx = logwx - logsumexp(logwx, 2);
                lognwx(isnan(lognwx)) = -inf;
                for i = 1:Nt
                    if all(isinf(lognwx(i, :)))
                    	k = randsample(1:Nx, 1, true);
                    else
                        k = randsample(1:Nx, 1, true, exp(lognwx(i, :)));
                    end
                    Xk(i, t) = Xm(i, k, t);
                end
            end
        end
    
    end
        
end
    
    

