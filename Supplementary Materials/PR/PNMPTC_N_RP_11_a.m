clc;clear;
%%
rng(20);
m = 3000; n = 10;
X = 0.8*randn(m,n);
x = [0.5; -1.2; 0.8; 1;-0.3;0.7;0.9;0.3;-0.1;-0.5;0.6];
X_with_int = [ones(m,1), X];
eta = X_with_int * x;
y = poissrnd(exp(eta));
%%
x0 = zeros(n+1,1);
%%
alpha = 0.01;
F = @(x)  -(1/m)*sum(y .* eta - exp(eta)) + alpha * norm(x,1);
%%
tspan = [0 1];
ns = zeros(n+1,1)+100; 
[t1, x1] = ode23(@(t, x) PTDDN_dxdt(t, x,-(1/m)*X_with_int' * (y - exp(X_with_int * x)),ns),tspan, x0);
%%
ns = @(x)zeros(n+1,1)+100*sin(x); 
[t2, x2] = ode23(@(t, x) PTDDN_dxdt2(t, x,-(1/m)*X_with_int' * (y - exp(X_with_int * x)),ns),tspan, x0);
ns = @(x)zeros(n+1,1)+100*x;
[t3, x3] = ode23(@(t, x) PTDDN_dxdt2(t, x,-(1/m)*X_with_int' * (y - exp(X_with_int * x)),ns),tspan, x0);
%%
mu = 0;  
gamma = 1;   
rng(3);
u = rand(n+1, 1);
cauchyNoise = mu + gamma * tan(pi * (u - 0.5));
ns = cauchyNoise; 
[t4, x4] = ode23(@(t, x) PTDDN_dxdt(t, x,-(1/m)*X_with_int' * (y - exp(X_with_int * x)),ns),tspan, x0);
%%
ns = randn(n+1,1); 
tic
[t5, x5] = ode23(@(t, x) PTDDN_dxdt(t, x,-(1/m)*X_with_int' * (y - exp(X_with_int * x)),ns),tspan, x0);
time = toc
%%
%¼ÆËãx-x*
cc1=[];cc2=[];cc3=[];cc4=[];cc5=[];
for j = 1:(length(t1))
        cc1(j)= log10(norm(x1(j,:)'-x)^2);
end
for j = 1:(length(t2))
        cc2(j)= log10(norm(x2(j,:)'-x)^2);
end
for j = 1:(length(t3))
        cc3(j)= log10(norm(x3(j,:)'-x)^2);
end
for j = 1:(length(t4))
        cc4(j)= log10(norm(x4(j,:)'-x)^2);
end
for j = 1:(length(t5))
        cc5(j)= log10(norm(x5(j,:)'-x)^2);
end
%%

figure
plot3(t1,ones(size(t1)),cc1,'Color', [0.90 0.29 0.23], 'Marker', 'v', 'LineStyle', '-', 'MarkerSize', 5)
hold on
plot3(t2,2*ones(size(t2)),cc2,'Color', [0.27 0.42 0.81], 'Marker', 'h', 'LineStyle', '-', 'MarkerSize', 5)
hold on
plot3(t3,3*ones(size(t3)),cc3,'Color', [0.96 0.73 0.12], 'Marker', 'p', 'LineStyle', '-', 'MarkerSize', 5)
hold on
plot3(t4,4*ones(size(t4)),cc4,'Color', [0.57 0.27 0.67], 'Marker', 'o', 'LineStyle', '-', 'MarkerSize', 5)
hold on
plot3(t5,5*ones(size(t5)),cc5,'Color',  [0.6 0 0.3], 'Marker', '>', 'LineStyle', '-', 'MarkerSize', 5)
xlabel('t');
ylabel('Noise');
zlabel('log_{10}||x-x^*||^2');
legend('r=[100;...;100]','r=[100sin(t);...;100sin(t)]','r=[100t;...;100t]','r=Cauchy distribution noise','r=Gaussian noise');
grid on;
view(45, 20); 

%%
function x_t = PTDDN_dxdt(t,x,grad,N)
alpha = 0.01; 
lambda = 0.6; 
Tp = 1;
r = 1;
proximal = @(x_new, da) sign(x_new) .* max(abs(x_new) - da, 0);
u = @(t) r/((Tp-t)^1);
    if t <Tp-1e-12
         x_t = -u(t)* (x-proximal(x - lambda * grad, lambda * alpha))+N;
    else
        x_t = -(1/(1e-12)^1)  * (x-proximal(x - lambda * grad, lambda * alpha))+N;
    end
end
%%
function x_t = PTDDN_dxdt2(t,x,grad,ns)
alpha = 0.01;
lambda = 0.6; 
Tp = 1;
r = 1;
proximal = @(x_new, da) sign(x_new) .* max(abs(x_new) - da, 0);
u = @(t) r/((Tp-t)^1);
    if t <Tp-1e-12
         x_t = -u(t)* (x-proximal(x - lambda * grad, lambda * alpha))+ns(t);
    else
        x_t = -(1/(1e-12)^1)  * (x-proximal(x - lambda * grad, lambda * alpha))+ns(t);
    end
end

