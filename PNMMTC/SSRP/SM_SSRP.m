clc; clear;
%SSRP_1.mat-->R^{4*8}
%SSRP_2.mat-->R^{256*1024}
%SSRP_3.mat-->R^{512*2048}
data = load('SSRP_1.mat');
A = data.A;
b = data.b;
x_true = data.x;  
n = size(A,2);
lambda = 0.001;     
max_iter = 20000;    
tol = 1e-8;           
R=max(eig(A'*A));
tic;
x = zeros(n,1);       
obj_vals = zeros(max_iter,1);
err_vals = zeros(max_iter,1);
for k = 1:max_iter
    grad = A' * (A*x - b) + lambda * sign(x);
    t=10/sqrt(k);
    x_new = x - t * grad;
    obj_vals(k) = 0.5 * norm(A*x_new - b)^2 + lambda * norm(x_new, 1);
    err_vals(k) = log10(norm(x_new - x_true)^2);
    if norm(x_new - x) < tol
        fprintf('SM %d \n', k);
        break;
    end
    x = x_new;
end
time = toc
fprintf('erorr ||x-x^*||^2 = %.4e\n', norm(x - x_true)^2);