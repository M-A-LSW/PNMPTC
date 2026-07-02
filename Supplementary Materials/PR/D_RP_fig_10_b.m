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
tau = 4;         
alpha_delay = 4;
lags = [tau, alpha_delay];
%% 
grad = @(x) -(1/m)*X_with_int' * (y - exp(X_with_int * x));
prox = @(v, lambda) sign(v) .* max(abs(v) - lambda, 0);
P_func_general = @(x, lambda, alpha) x - prox(x - lambda * grad(x), lambda * alpha);

%% ===================== PNA =====================
lambda1 = 0.6;
alpha_reg1 = 0.01;
P1 = @(x) P_func_general(x, lambda1, alpha_reg1);
x0 = zeros(n+1, 1);
history1 = @(t) [x0; alpha_delay * P1(x0)];
gain1 = @(t, x) 40;
ode1 = @(t, y, Z) delay_model_full(t, y, Z, gain1, P1);

%% ===================== FNPGNN =====================
lambda2 = 0.6;
alpha_reg2 = 0.01;
A = 10; R1 = 0.5;
P2 = @(x) P_func_general(x, lambda2, alpha_reg2);
history2 = @(t) [x0; alpha_delay * P2(x0)];
gain2 = @(t, x) A / (norm(P2(x))^(1-R1));
ode2 = @(t, y, Z) delay_model_full(t, y, Z, gain2, P2);

%% ===================== FXPNA =====================
lambda3 = 0.6;
alpha_reg3 = 0.01;
A3 = 10; B3 = 10; C3 = 20; R1_3 = 0.5; R2_3 = 1.5;
P3 = @(x) P_func_general(x, lambda3, alpha_reg3);
history3 = @(t) [x0; alpha_delay * P3(x0)];
gain3 = @(t, x) A3 / (norm(P3(x))^(1-R1_3)) + B3 / (norm(P3(x))^(1-R2_3)) + C3;
ode3 = @(t, y, Z) delay_model_full(t, y, Z, gain3, P3);

%% =====================TvNOA =====================
lambda4 = 0.6;
alpha_reg4 = 0.01;
P4 = @(x) P_func_general(x, lambda4, alpha_reg4);
history4 = @(t) [x0; alpha_delay * P4(x0)];
w=0.1;
A4=10;B4=10;
gain4 = @(t, x)  (A4/(t+w))/ (norm(P4(x))^(1-R1_3)) +( B4/(t+w)) / (norm(P4(x))^(1-R2_3));
ode4 = @(t, y, Z) delay_model_full(t, y, Z, gain4, P4);

%% =====================PTTVCN =====================
lambda5 = 0.6;
alpha_reg5 = 0.01;
L = 1; st = 0.82; 
c1=10;c2=10;w=0.1;
Tp = 1; 
Vk=sqrt(1-2*lambda5*st+(lambda5^2)*(L^2));
Wk=sqrt(1-2*lambda5*st+(lambda5^2)*(L^2));
Ck1 = (1+R1_3)/2; Ck2 = (1+R2_3)/2;
a1 = ((2^Ck1)*(1-Vk))/Wk^(1-R1_3);
a2 = (2^Ck2)*(1-Vk)^R2_3;
Pp =(Tp*(1/(c1*a1*(1-Ck1))+1/(c2*a2*(Ck2-1))))/log((w+Tp)/w);
P5 = @(x) P_func_general(x, lambda5, alpha_reg5);
history5 = @(t) [x0; alpha_delay * P5(x0)];
gain5 = @(t, x) (Pp/Tp) * ((c1/(t+w))/(norm(P4(x))^(1-R1_3)) + (c2/(t+w))/(norm(P4(x))^(1-R2_3)));
ode5 = @(t, y, Z) delay_model_full(t, y, Z, gain5, P5);

%% ===================== PNMPTC =====================
% 
lambda6 = 0.6;
alpha_reg6 = 0.01;
P6 = @(x) P_func_general(x, lambda6, alpha_reg6);
history6 = @(t) [x0; alpha_delay * P6(x0)];
Tp=1;
ode6 = @(t, y, Z) delay_model_split(t, y, Z,Tp, P6);

%% 
models = {'PNA', 'FNPGNN', 'FXPNA', 'TvNOA', 'PTTVCN', 'PNMPTC'};
odes = {ode1, ode2, ode3, ode4, ode5, ode6};
histories = {history1, history2, history3, history4, history5, history6};
sol_cell = cell(6,1);
tspan = [0, 5];  
for i = 1:4
    fprintf(' %s...\n', models{i});
    tic;
    sol_cell{i} = dde23(odes{i}, lags, histories{i}, tspan);
    toc;
end
fprintf('%s...\n', models{5});
tic;
tspan = [0, 1];  
sol_cell{5} = dde23(odes{5}, lags, histories{5}, tspan);
toc;
fprintf('%s...\n', models{6});
tic;
sol_cell{6} = dde23(odes{6}, lags, histories{6}, tspan);
toc;
%% 
t_all = cell(6,1);
x_all = cell(6,1);
err_log_all = cell(6,1);  % log10(||x-x_true||^2)

for i = 1:6
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
figure;
for i = 1:6
    plot(t_all{i}, err_log_all{i}, 'Color', colors(i,:), 'Marker', markers{i}, 'LineStyle', '-', 'MarkerSize', 5);
    hold on;
end
xlabel('t'); ylabel('log_1_0||x-x*||^2');
legend(models, 'Location', 'best');


%%
% ode：
%   dx/dt = - gain(t,x) * P(x) + P(x(t-τ)) + I
%   dI/dt = P(x) - P(x(t-α))
function dydt = delay_model_full(t, y, Z, gain_func, P_func)
    x = y(1:end/2);
    I = y(end/2+1:end);
    x_tau = Z(1:end/2, 1);
    x_alpha = Z(1:end/2, 2);
    Px = P_func(x);
    Px_tau = P_func(x_tau);
    Px_alpha = P_func(x_alpha);
    dxdt = - gain_func(t, x) * Px + Px_tau + I;
    dIdt = Px - Px_alpha;
    dydt = [dxdt; dIdt];
end

%%
function dydt = delay_model_split(t, y, Z,  Tp, P_func)
    r = 1;
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
