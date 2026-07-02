clc;clear;
%%
data = load('SSRP_1.mat');
A = data.A; b = data.b; x = data.x;
[~,n]=size(A);
%%
alpha = 0.001; 

F = @(x) norm(A*x-b)^2 + alpha*norm(x,1);
%%
Tp=1;
tspan = [0 Tp];
x0= zeros(n, 1);
tic;
[t1, x1] = ode23(@(t, x) PNMPTC_dxdt(t, x,A,b,A'*(A * x - b),n,Tp),tspan, x0);%grad实时变化
time=toc
% filename = 't.mat'; 
% save(filename, 't1');
%%
cc1=[];
for j = 1:(length(t1))
        cc1(j)= log10(norm(x1(j,:)'-x)^2);
end
fprintf('PNMPTC ||x-x^*||^2 = %.4e\n',norm(x1(end,:)'-x)^2);
%%
figure
plot(t1,cc1,'Color', [0.90 0.29 0.23], 'Marker', 'v', 'LineStyle', '-', 'MarkerSize', 4)
xlabel('t');
ylabel('x(t)');
legend('PNMPTC');
%%
function x_t = PNMPTC_dxdt(t,x,A,b,grad,n,Tp)
alpha = 0.001;
lambda = 1.5;
r = 1;
k = 1e-12;
proximal = @(x_new, da) sign(x_new) .* max(abs(x_new) - da, 0);
u = @(t) r/((Tp-t)^1.2);
um= r/((k)^1.2);
    if t <Tp-k
         x_t = -u(t)* (x-proximal(x - lambda * grad, lambda * alpha));
    else
        x_t = -um * (x-proximal(x - lambda * grad, lambda * alpha));
    end
end

