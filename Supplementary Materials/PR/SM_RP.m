clc; clear;
rng(20);
m = 3000; n = 10;
X = 0.8 * randn(m, n);
x_true = [0.5; -1.2; 0.8; 1; -0.3; 0.7; 0.9; 0.3; -0.1; -0.5; 0.6];
X_with_int = [ones(m, 1), X];
eta = X_with_int * x_true;
y = poissrnd(exp(eta));
lambda = 0.01;     
max_iter = 20000;    
tol = 1e-8;           
tic;
x = zeros(n+1,1);       
F = @(x)  -(1/m)*sum(y .* eta - exp(eta)) + lambda * norm(x,1);
obj_vals = zeros(max_iter,1);
err_vals = zeros(max_iter,1);
for k = 1:max_iter
    grad = -(1/m)*X_with_int' * (y - exp(X_with_int * x)) + lambda * sign(x);
    t=0.1/sqrt(k);
    x_new = x - t * grad;
    obj_vals(k) = F(x);
    err_vals(k) = log10(norm(x_new - x_true)^2);
    if norm(x_new - x) < tol
        fprintf('SM %d \n', k);
        break;
    end
    x = x_new;
end
time = toc
fprintf('erorr ||x-x^*||^2 = %.4e\n', norm(x - x_true)^2);