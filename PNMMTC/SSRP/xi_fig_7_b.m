clc;clear;
%%\xi
data = load('SSRP_1.mat');
A = data.A; b = data.b; x = data.x;
n=8;
%%
alpha = 0.001;
F = @(x) norm(A*x-b)^2 + alpha*norm(x,1);
%%
tspan = [0 1];
x0= zeros(n, 1);
tic
T=1.15;
[t1, x1] = ode23(@(t, x) PTDDN_dxdt(t, x,A,b,A'*(A * x - b),n,T),tspan, x0);
te1=toc
%%
tic
T=1.2;
[t2, x2] = ode23(@(t, x) PTDDN_dxdt(t, x,A,b,A'*(A * x - b),n,T),tspan, x0);
te2=toc
%%
tic
T=1.3;
[t3, x3] = ode23(@(t, x) PTDDN_dxdt(t, x,A,b,A'*(A * x - b),n,T),tspan, x0);
te3=toc
%%
tic
T=1.5;
[t4, x4] = ode23(@(t, x) PTDDN_dxdt(t, x,A,b,A'*(A * x - b),n,T),tspan, x0);
te4=toc
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
figure;
xi= [1.15, 1.2, 1.3, 1.5];
plot3(t1, xi(1)*ones(size(t1)),cc1,  'Color', [0.90 0.29 0.23], 'Marker', 'v', 'LineStyle', '-', 'MarkerSize', 4);
hold on;
plot3(t2, xi(2)*ones(size(t2)),cc2, 'Color', [0.27 0.42 0.81], 'Marker', 'h', 'LineStyle', '-', 'MarkerSize', 4);
plot3(t3, xi(3)*ones(size(t3)),cc3,  'Color', [0.96 0.73 0.12], 'Marker', 'p', 'LineStyle', '-', 'MarkerSize', 4);
plot3(t4, xi(4)*ones(size(t4)),cc4,  'Color', [0.57 0.27 0.67], 'Marker', 'o', 'LineStyle', '-', 'MarkerSize', 4);
hold off;
xlabel('t');
ylabel('\xi');
zlabel('log_{10}||x-x^*||^2');
legend('\xi=1.15','\xi=1.2','\xi=1.3','\xi=1.5', 'Location','best');
grid on;
view(45, 20); 
set(gca, 'YTick', xi);
%%
function x_t = PTDDN_dxdt(t,x,A,b,grad,n,td)
alpha = 0.001; 
lambda = 1.5; 
Tp = 1;
r = 1;
proximal = @(x_new, da) sign(x_new) .* max(abs(x_new) - da, 0);
u = @(t) r/((Tp-t)^td);
    if t <Tp-1e-12
         x_t = -u(t)* (x-proximal(x - lambda * grad, lambda * alpha));
    else
        x_t = -(1/(1e-12)^td)  * (x-proximal(x - lambda * grad, lambda * alpha));
    end
end


