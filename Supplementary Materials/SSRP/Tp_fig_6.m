clc;clear;
%% T_p
data = load('SSRP_1.mat');
A = data.A; b = data.b; x = data.x;
n=8;
%%
alpha = 0.001; 
F = @(x) norm(A*x-b)^2 + alpha*norm(x,1);
x0= zeros(n, 1);
%%
tspan = [0 0.1];
T=0.1;
[t1, x1] = ode23(@(t, x) PTDDN_dxdt(t, x,A,b,A'*(A * x - b),n,T),tspan, x0);
%%
tspan = [0 0.5];
T=0.5;
[t2, x2] = ode23(@(t, x) PTDDN_dxdt(t, x,A,b,A'*(A * x - b),n,T),tspan, x0);
%%
tspan = [0 1];
T=1;
[t3, x3] = ode23(@(t, x) PTDDN_dxdt(t, x,A,b,A'*(A * x - b),n,T),tspan, x0);
%%
tspan = [0 2];
T=2;
[t4, x4] = ode23(@(t, x) PTDDN_dxdt(t, x,A,b,A'*(A * x - b),n,T),tspan, x0);
%%
cc1=[];cc2=[];cc3=[];cc4=[];
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
%%
figure
plot(0,0,'Color', [0.90 0.29 0.23], 'Marker', 'v','MarkerSize', 4)
hold on;
plot(0,0,'Color', [0.27 0.42 0.81], 'Marker', 'h',  'MarkerSize', 4)
hold on;
plot(0,0,'Color', [0.96 0.73 0.12], 'Marker', 'p',  'MarkerSize', 4)
hold on;
plot(0,0,'Color', [0.57 0.27 0.67], 'Marker', 'o',  'MarkerSize', 4)
hold on;
plot(0.1,x,'k*', 'LineWidth', 1, 'MarkerSize', 8)%, 'LineWidth', 2, 'MarkerSize', 10
hold on;
plot(t1,x1,'Color', [0.90 0.29 0.23], 'Marker', 'v', 'LineStyle', '-', 'MarkerSize', 4)
hold on;
plot(t2,x2,'Color', [0.27 0.42 0.81], 'Marker', 'h', 'LineStyle', '-', 'MarkerSize', 4)
hold on;
plot(t3,x3,'Color', [0.96 0.73 0.12], 'Marker', 'p', 'LineStyle', '-', 'MarkerSize', 4)
hold on;
plot(t4,x4,'Color', [0.57 0.27 0.67], 'Marker', 'o', 'LineStyle', '-', 'MarkerSize', 4)
hold on;
plot(0.1,x,'k*', 'LineWidth', 1, 'MarkerSize', 8)
hold on
plot(0.5,x,'k*', 'LineWidth', 1, 'MarkerSize', 8)%, 'LineWidth', 2, 'MarkerSize', 10
hold on
plot(1,x,'k*', 'LineWidth', 1, 'MarkerSize', 8)%, 'LineWidth', 2, 'MarkerSize', 10
hold on
plot(2,x,'k*', 'LineWidth', 1, 'MarkerSize', 8)%, 'LineWidth', 2, 'MarkerSize', 10
xlabel('t');
ylabel('x(t)');
legend('T_p=0.1','T_p=0.5','T_p=1','T_p=2','x*');
%print('-depsc2', '-r600', 'SSRPTP1.eps');

figure
plot(t1,cc1,'Color', [0.90 0.29 0.23], 'Marker', 'v', 'LineStyle', '-', 'MarkerSize', 4)
hold on
plot(t2,cc2,'Color', [0.27 0.42 0.81], 'Marker', 'h', 'LineStyle', '-', 'MarkerSize', 4)
hold on
plot(t3,cc3,'Color', [0.96 0.73 0.12], 'Marker', 'p', 'LineStyle', '-', 'MarkerSize', 4)
hold on
plot(t4,cc4,'Color', [0.57 0.27 0.67], 'Marker', 'o', 'LineStyle', '-', 'MarkerSize', 4)
xlabel('t');
ylabel('log_1_0||x-x*||^2');
legend('T_p=0.1','T_p=0.5','T_p=1','T_p=2');
%print('-depsc2', '-r600', 'SSRPTP2.eps');
%%
function x_t = PTDDN_dxdt(t,x,A,b,grad,n,T)
alpha = 0.001; 
lambda = 0.85; 
Tp = T;
r = 1;
proximal = @(x_new, da) sign(x_new) .* max(abs(x_new) - da, 0);
u = @(t) r/((Tp-t)^1.2);
    if t <Tp-1e-12
         x_t = -u(t)* (x-proximal(x - lambda * grad, lambda * alpha));
    else
        x_t = -(1/(1e-12)^1.2)  * (x-proximal(x - lambda * grad, lambda * alpha));
    end
end

