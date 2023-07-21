%% Brownian motion model

rng(1)
T = 100; 
theta = [1 1.2 log(1.5) log(1)]; 
m = BrownianMotion_model(T, 'theta', theta);
figure; plot(1:m.T, m.y)

save('bm_model.mat', 'm')

%% Flexible-Allee model 

load('data_nutria.mat', 'y')
T = 60;
y = y(1:T);
m = FlexibleAllee_model(T, 'y', log(y)); 
figure; plot(1:m.T, m.y, '-');

save('fa_model.mat', 'm');

%% Stochastic Volatility in mean weekly

Data = readtable('^GSPTSE_weekly.csv');
log_Pt = log(Data.Close);
dlog_Pt = log_Pt(2:end) - log_Pt(1:end-1);
y = 100*(dlog_Pt  - mean(dlog_Pt));

T = length(y)-1;
m = StochasticVolatilityInMean_model(T, y(1), 'y', y(2:end));
figure; plot(1:m.T, m.y);

save('svmw_model.mat', 'm');

%% Stochastic Volatility in mean daily 

Data = readtable('^GSPTSE_daily.csv');
log_Pt = log(Data.Close);
dlog_Pt = log_Pt(2:end) - log_Pt(1:end-1);
y_full = 100*(dlog_Pt  - mean(dlog_Pt));

T = (length(y_full)-1)/2;
y = y_full(1:T+1);

m = StochasticVolatilityInMean_model(T, y(1), 'y', y(2:end));
figure; plot(1:m.T, m.y);

save('svmd_model.mat', 'm');

%% Univariate Ornstein-Uhlenbeck model

rng(1)
T = 400; 
dbeta = 20;
phi = 0.5;
theta = [log(phi/(1-phi)) 0.38 log(1) repmat(0.1, 1, dbeta)]; 
m = UnivariateOU_model(T, dbeta, 'theta', theta);
figure; plot(1:m.T, m.y)

save('ou_model.mat', 'm')
