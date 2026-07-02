clc; clear;
%%
rng(20);
m = 3000; n = 10;
X = 0.8*randn(m,n);
x = [0.5; -1.2; 0.8; 1;-0.3;0.7;0.9;0.3;-0.1;-0.5;0.6];
X_with_int = [ones(m,1), X];
eta = X_with_int * x;
y = poissrnd(exp(eta));
%% 
grad = @(x) - (1/m) * X_with_int' * (y - exp(X_with_int * x));
prox = @(v, lambda) sign(v) .* max(abs(v) - lambda, 0);
P_func_general = @(x, lambda, alpha) x - prox(x - lambda * grad(x), lambda * alpha);

%% ===================== PNMPTC =====================
lambda = 0.6;
alpha_reg = 0.01;
x0 = zeros(n+1, 1);
P = @(x) P_func_general(x, lambda, alpha_reg);
Tp=1; 
%%
tau1 = 0.5;          
alpha_delay1 = 0.6;
lags1 = [tau1, alpha_delay1];
history1 = @(t) [x0; alpha_delay1 * P(x0)];
ode1 = @(t, y, Z) delay_model_split(t, y, Z, Tp, P);
%%
tau2 = 0.6;          
alpha_delay2 = 0.5;
lags2 = [tau2, alpha_delay2];
history2 = @(t) [x0; alpha_delay2 * P(x0)];
ode2 = @(t, y, Z) delay_model_split(t, y, Z, Tp, P);
%%
alpha_delay3 = 0.7;
lags3 = @(t, y) [1/(t+1); alpha_delay3];
history3 = @(t) [x0; alpha_delay3 * P(x0)];
ode3 = @(t, y, Z) delay_model_split(t, y, Z, Tp, P);
%%
alpha_delay4 = 0.8;
lags4 = @(t, y) [1/(t+1)^2; alpha_delay4];
history4 = @(t) [x0; alpha_delay4 * P(x0)];
ode4 = @(t, y, Z) delay_model_split(t, y, Z, Tp, P);
%% 
models = {'\alpha=0.5,\tau(t)=0.6','\alpha=0.6,\tau(t)=0.5','\alpha=0.7,\tau(t)=1/(t+1)', '\alpha=0.8,\tau(t)=1/(t+1)^2'};
odes = {ode1, ode2, ode3, ode4};
lags = {lags1, lags2, lags3, lags4};
histories = {history1, history2, history3, history4};
sol_cell = cell(4,1);
tspan = [0, 1];  
for i = 1:2
    fprintf(' %s...\n', models{i});
    tic;
    sol_cell{i} = dde23(odes{i}, lags{i}, histories{i}, tspan);
    toc;
end
for i = 3:4
    fprintf(' %s...\n', models{i});
    tic;
    sol_cell{i} = ddesd(odes{i}, lags{i}, histories{i}, tspan);
    toc;
end
%%
t_all = cell(4,1);
x_all = cell(4,1);
err_log_all = cell(4,1);  

for i = 1:4
    sol = sol_cell{i};
    t_all{i} = sol.x;
    x_all{i} = sol.y(1:n+1, :);
    err = zeros(1, length(t_all{i}));
    for j = 1:length(t_all{i})
        err(j) = log10(norm(x_all{i}(:,j) - x)^2);
    end
    err_log_all{i} = err;
end

%% 
colors = [0.90 0.29 0.23; 0.27 0.42 0.81; 0.96 0.73 0.12; 0 0.6 0.3; 0.6 0 0.3; 0.57 0.27 0.67];
markers = {'v','h','p','s','>','o'};
alpha = [0.5 0.6 0.7 0.8];
figure;
for i = 1:4
    plot3(t_all{i}, i*ones(size(t_all{i})),err_log_all{i}, 'Color', colors(i,:), 'Marker', markers{i}, 'LineStyle', '-', 'MarkerSize', 4);
    hold on;
end
xlabel('t');
ylabel('Delay');
zlabel('log_{10}||x-x^*||^2');
legend(models, 'Location', 'best');
grid on;
view(45, 20); 
%%
function dydt = delay_model_split(t, y, Z, Tp, P_func)
    r=1;
    x = y(1:end/2);
    I = y(end/2+1:end);
    x_tau = Z(1:end/2, 1);
    x_alpha = Z(1:end/2, 2);
    Px = P_func(x);
    Px_tau = P_func(x_tau);
    Px_alpha = P_func(x_alpha);
    mu = @(t) r / (Tp - t)^1;
    if t < Tp - 1e-12
        dxdt = -mu(t) * Px + Px_tau + I;
    else
        dxdt = - (1/(1e-12)^1) * Px + Px_tau + I;
    end
    dIdt = Px - Px_alpha;
    dydt = [dxdt; dIdt];
end
