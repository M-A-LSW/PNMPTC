clc; clear;
%%
data = load('SSRP_1.mat');
A = data.A; 
b = data.b; 
x_true = data.x; 
n = 8;
%%
lambda = 0.001;  
omega = 1.5;  
Tp = 1;       
xi = 1.2;
delta=1;
x0 = zeros(n, 1);

