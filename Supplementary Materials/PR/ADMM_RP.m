clc; clear;
rng(20);
m = 3000; n = 10;
X = 0.8 * randn(m, n);
x_true = [0.5; -1.2; 0.8; 1; -0.3; 0.7; 0.9; 0.3; -0.1; -0.5; 0.6];
X_with_int = [ones(m, 1), X];
eta = X_with_int * x_true;
y = poissrnd(exp(eta));

lambda = 0.01;      
rho = 1;                
max_iter = 10000;
tol = 1e-8;

beta = zeros(n+1, 1);   
z = zeros(n+1, 1);
u = zeros(n+1, 1);

X = X_with_int;
[m, p] = size(X);
Atb = X' * y;            

err_vals = zeros(max_iter, 1);
obj_vals = zeros(max_iter, 1);

tic;
for k = 1:max_iter
    beta_old = beta;

    beta_sub = beta;      
    for newton = 1:20     
        mu = exp(min(X * beta_sub, 700));  
        grad_f = -(1/m) * X' * (y - mu);
        H = (1/m) * (X' * (mu .* X)) + rho * eye(p);
        grad = grad_f + rho * (beta_sub - z + u);
        delta = - H \ grad;
        t_step = 1;
        while t_step > 1e-12
            beta_new = beta_sub + t_step * delta;
            obj_new = -(1/m)*sum(y .* (X*beta_new) - exp(X*beta_new)) + ...
                      0.5*rho*norm(beta_new - z + u)^2;
            obj_old = -(1/m)*sum(y .* (X*beta_sub) - exp(X*beta_sub)) + ...
                      0.5*rho*norm(beta_sub - z + u)^2;
            if obj_new <= obj_old + 1e-4 * t_step * grad' * delta
                break;
            end
            t_step = t_step * 0.5;
        end
        beta_sub = beta_sub + t_step * delta;
    end
    beta = beta_sub;
    
    v = beta + u;
    z_old = z;
    z = sign(v) .* max(abs(v) - lambda/rho, 0);
    
    u = u + (beta - z);
    
    obj_vals(k) = -(1/m)*sum(y .* (X*beta) - exp(X*beta)) + lambda * norm(beta,1);
    err_vals(k) = log10(norm(beta - x_true)^2);
    
    if norm(z - z_old) < tol
        fprintf('ADMM %d\n', k);
        break;
    end
end
time = toc

fprintf('ADMM||x-x^*||^2 = %.4e\n', norm(beta - x_true)^2);