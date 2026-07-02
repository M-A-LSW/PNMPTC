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
rho = 2;          
max_iter = 10000;
tol = 1e-8;
x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);
obj_vals = zeros(max_iter,1);
err_vals = zeros(max_iter,1);
tic;
L = chol(A' * A + rho * eye(n), 'lower'); 
for k = 1:max_iter
    rhs = A' * b + rho * (z - u);
    x = L' \ (L \ rhs);          
    v = x + u;
    zk=z;
    z = sign(v) .* max(abs(v) - lambda/rho, 0);
    u = u + (x - z);
    obj_vals(k) = 0.5 * norm(A*x - b)^2 + lambda * norm(x,1);
    err_vals(k) = log10(norm(x - x_true)^2);
 
    if norm(zk - z) < tol
        fprintf('ADMM = %d\n', k);
        break;
    end
end
time=toc
fprintf('V=||x-x^*||^2 = %.4e\n', norm(x - x_true)^2);