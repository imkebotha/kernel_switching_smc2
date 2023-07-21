%% load particular model

% Brownian motion 
load('bm_model.mat', 'm')
Nx = 200;

% Flexible-allee
load('fa_model.mat', 'm')
m.theta = [6.289329170623962   0.017704571934860   0.000028183860786...
    -0.000000008073341  -2.442379163725157  -4.836418226194660];
Nx = 1700;

% SVM weekly
load('svmw_model.mat', 'm')
m.theta = [0.3717   -4.0414   -0.1264    0.3904    2.6435   -1.3360];
Nx = 60;

% SVM daily
load('svmd_model.mat', 'm')
m.theta = [0.0945   -2.6808   -0.1203   -0.2128    3.7320   -1.8269];
Nx = 200;

% OU 
load('ou_model.mat', 'm')
Nx = 420;

%% Run a particle MCMC sampler

Nt = 10000;
Sig = 0.1*eye(m.np);
MALA_stepsize = 0.1; MALA_ar_target = 0.574;

% PMMH
pmmhobj = MCMC_sampler(m, Sig, m.theta);
pmmhobj.sample(Nt, Nx);
plot_samples(m, pmmhobj.tsamples, 'TracePlots', true)

% PG
pgobj = MCMC_PG_sampler(m, Sig, m.theta);
pgobj.sample(Nt, Nx, MALA_stepsize);
plot_samples(m, pgobj.tsamples, 'TracePlots', true)

% tune parameters
Sig = cov(pmmhobj.tsamples); 
r = pgobj.acc/MALA_ar_target; 
MALA_stepsize = MALA_stepsize * exp(2*r-2);

%% SMC^2

Nt = 100;

% method = "DataAnnealing"; 
method = "DensityTempering"; 

% default kernel
% def_kernel = "pmmh";
def_kernel = "pg";

% switch kernels
kswitch = "always";
% kswitch = "lag";
% kswitch = "none";

% fraction of Nx to use for PG
r = 1;

smc2obj = SMC2(m, method, 'kernel', def_kernel, 'kernelAdaptation', kswitch); 
smc2obj.sample(Nt, Nx, r);

% view results
smc2obj.results_summary
plot_samples(m, smc2obj.tsamples, 'Reference', pmmhobj);

