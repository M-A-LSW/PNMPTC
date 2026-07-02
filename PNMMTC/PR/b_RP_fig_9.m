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
%TvNOA
L = 1;
lambda = 0.6;
c1=10;
c2=10;
w=0.1;
R_da1 = 0.5;
R_da2 = 1.5;
tspan = [0 5];
proximal = @(x_new, da) sign(x_new) .* max(abs(x_new) - da, 0);
tic;
dxdt1 = @(t, x,grad) -((c1/(t^2+w))/(norm(x-proximal(x - lambda * grad, lambda * alpha)))^(1-R_da1)+(c2/(t^2+w))/norm(x-proximal(x - lambda * grad, lambda * alpha))^(1-R_da2))*(x-proximal(x - lambda * grad, lambda * alpha));
[t3, x3] = ode23(@(t, x) dxdt1(t, x,-(1/m)*X_with_int' * (y - exp(X_with_int * x))),tspan, x0);
time1=toc
fprintf('TvNOA ||x-x^*||^2 = %.4e\n',norm(x3(end,:)'-x)^2);
%%
%PNA
dxdt2 = @(t, x,grad) -40*(x-proximal(x - lambda * grad, lambda * alpha));
tic;
[t1, x1] = ode23(@(t, x) dxdt2(t, x,-(1/m)*X_with_int' * (y - exp(X_with_int * x))),tspan, x0);
time2= toc
fprintf('PNA ||x-x^*||^2 = %.4e\n',norm(x1(end,:)'-x)^2);
%%
%FXPNA
L = 1;
A_da = 10;
B_da = 10;
C_da = 10;
R_da1 = 0.5;
R_da2 = 1.5;
dxdt3 = @(t, x,grad) -(A_da/(norm(x-proximal(x - lambda * grad, lambda * alpha)))^(1-R_da1)+B_da/norm(x-proximal(x - lambda * grad, lambda * alpha))^(1-R_da2)+C_da)*(x-proximal(x - lambda * grad, lambda * alpha)) ;
tic;
[t2, x2] = ode23(@(t, x) dxdt3(t, x,-(1/m)*X_with_int' * (y - exp(X_with_int * x))),tspan, x0);
time3= toc
fprintf('PXPNA||x-x^*||^2 = %.4e\n',norm(x2(end,:)'-x)^2);
%%
%PNMPTC
tic
tspan = [0 1];
[t4, x4] = ode23(@(t, x) PTDDN_dxdt(t, x,-(1/m)*X_with_int' * (y - exp(X_with_int * x))),tspan, x0);
time4 = toc
fprintf('PNMPTC ||x-x^*||^2 = %.4e\n',norm(x4(end,:)'-x)^2);
%%
%FNPGNN
L = 1;
A_da = 10;
R_da1 = 0.5;
tspan = [0 5];
dxdt5 = @(t, x,grad) -(A_da/(norm(x-proximal(x - lambda * grad, lambda * alpha)))^(1-R_da1))*(x-proximal(x - lambda * grad, lambda * alpha)) ;
tic;
[t5, x5] = ode23(@(t, x) dxdt5(t, x,-(1/m)*X_with_int' * (y - exp(X_with_int * x))),tspan, x0);
time5= toc
fprintf('PNPGNN||x-x^*||^2 = %.4e\n',norm(x5(end,:)'-x)^2);
%%
%PTTVCN
alpha = 0.01; 
lambda = 0.6;
L = 1;
c1=10;
c2=10;
w=0.1;
R_da1 = 0.5;
R_da2 = 1.5;
st = 0.82;
Tp = 1;
Vk=sqrt(1-2*lambda*st+(lambda^2)*(L^2));
Wk=sqrt(1-2*lambda*st+(lambda^2)*(L^2));
Ck1=(1+R_da1)/2;
Ck2=(1+R_da2)/2;
a1 =((2^Ck1)*(1-Vk))/Wk^(1-R_da1);
a2 =(2^Ck2)*(1-Vk)^R_da2;
Pp =Tp*sqrt(w)*(1/(c1*a1*(1-Ck1))+1/(c2*a2*(Ck2-1)))/atan(Tp/sqrt(w));
tspan = [0 1];
proximal = @(x_new, da) sign(x_new) .* max(abs(x_new) - da, 0);
dxdt6 = @(t, x,grad) -(Pp/Tp)*((c1/(t^2+w))/(norm(x-proximal(x - lambda * grad, lambda * alpha)))^(1-R_da1)+(c2/(t^2+w))/norm(x-proximal(x - lambda * grad, lambda * alpha))^(1-R_da2))*(x-proximal(x - lambda * grad, lambda * alpha));
tic;
[t6, x6] = ode23(@(t, x) dxdt6(t, x,-(1/m)*X_with_int' * (y - exp(X_with_int * x))),tspan, x0);
time6= toc
fprintf('PTTVCN ||x-x^*||^2 = %.4e\n',norm(x6(end,:)'-x)^2);
%%
cc1=[];cc2=[];cc3=[];cc4=[];cc5=[];cc6=[];
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
for j = 1:(length(t6))
        cc6(j)= log10(norm(x6(j,:)'-x)^2);
end
%%
figure
plot(0,0,'Color', [0.90 0.29 0.23], 'Marker', 'v','MarkerSize', 4)
hold on;
plot(0,0,'Color', [0.27 0.42 0.81], 'Marker', 'h',  'MarkerSize', 4)
hold on;
plot(0,0,'Color', [0.96 0.73 0.12], 'Marker', 'p',  'MarkerSize', 4)
hold on;
plot(0,0,'Color', [0 0.6 0.3], 'Marker', 's',  'MarkerSize', 4)
hold on;
plot(0,0,'Color', [0.6 0 0.3], 'Marker', '>',  'MarkerSize', 4)
hold on;
plot(0,0,'Color', [0.57 0.27 0.67], 'Marker', 'o',  'MarkerSize', 4)
hold on;
plot(1,x,'k*', 'LineWidth', 1, 'MarkerSize', 8)%, 'LineWidth', 2, 'MarkerSize', 10
hold on;
plot(t1,x1,'Color', [0.90 0.29 0.23], 'Marker', 'v', 'LineStyle', '-', 'MarkerSize', 4)
hold on;
plot(t5,x5,'Color', [0.27 0.42 0.81], 'Marker', 'h', 'LineStyle', '-', 'MarkerSize', 4)
hold on;
plot(t2,x2,'Color', [0.96 0.73 0.12], 'Marker', 'p', 'LineStyle', '-', 'MarkerSize', 4)
hold on
plot(t3,x3,'Color', [0 0.6 0.3], 'Marker', 's', 'LineStyle', '-', 'MarkerSize', 4)
hold on;
plot(t6,x6,'Color', [0.6 0 0.3], 'Marker', '>', 'LineStyle', '-', 'MarkerSize', 4)
hold on;
plot(t4,x4,'Color', [0.57 0.27 0.67], 'Marker', 'o', 'LineStyle', '-', 'MarkerSize', 4)
hold on
plot(1,x,'k*', 'LineWidth', 1, 'MarkerSize', 8)
hold on
plot(5,x,'k*', 'LineWidth', 1, 'MarkerSize', 8)%, 'LineWidth', 2, 'MarkerSize', 10
xlabel('t');
ylabel('x(t)');
ylim([-1.3,1.1])
legend('PNA','FNPGNN','FXPNA','TvNOA','PTTVCN','PNMPTC','x*');
%print('-depsc2', '-r600', 'SSRPdb1_1.eps');

figure
plot(t1,cc1,'Color', [0.90 0.29 0.23], 'Marker', 'v', 'LineStyle', '-', 'MarkerSize', 4)
hold on
plot(t5,cc5,'Color', [0.27 0.42 0.81], 'Marker', 'h', 'LineStyle', '-', 'MarkerSize', 4)
hold on
plot(t2,cc2,'Color', [0.96 0.73 0.12], 'Marker', 'p', 'LineStyle', '-', 'MarkerSize', 4)
hold on
plot(t3,cc3,'Color', [0 0.6 0.3], 'Marker', 's', 'LineStyle', '-', 'MarkerSize', 4)
hold on
plot(t6,cc6,'Color', [0.6 0 0.3], 'Marker', '>', 'LineStyle', '-', 'MarkerSize', 4)
hold on
plot(t4,cc4,'Color', [0.57 0.27 0.67], 'Marker', 'o', 'LineStyle', '-', 'MarkerSize', 4)
xlabel('t');
ylabel('log_1_0||x-x*||^2');
legend('PNA','FNPGNN','FXPNA','TvNOA','PTTVCN','PNMPTC');


figure
plot(x1(end,:),'v','Color', [0.90 0.29 0.23], 'MarkerSize', 5)
hold on
plot(x5(end,:), 'h', 'Color', [0.27 0.42 0.81],  'MarkerSize', 5)
hold on
plot(x2(end,:),'p', 'Color', [0.96 0.73 0.12], 'MarkerSize', 5)
hold on
plot(x3(end,:), 's', 'Color', [0 0.6 0.3], 'MarkerSize', 5)
hold on
plot(x6(end,:),'>', 'Color', [0.6 0 0.3], 'MarkerSize', 5)
hold on
plot(x4(end,:), 'o', 'Color', [0.57 0.27 0.67], 'MarkerSize', 5)
hold on
plot(x,'k*')
xlabel('n');
ylabel('x');
legend('PNA','FNPGNN','FXPNA','TvNOA','PTTVCN','PNMPTC','x*');

%%
function x_t = PTDDN_dxdt(t,x,grad)
alpha = 0.01;
lambda = 0.6;
Tp = 1;
r = 1;
proximal = @(x_new, da) sign(x_new) .* max(abs(x_new) - da, 0);
u = @(t) r/((Tp-t)^1);
    if t <Tp-1e-11
         x_t = -u(t)* (x-proximal(x - lambda * grad, lambda * alpha));
    else
        x_t = -(1/(1e-11)^1)  * (x-proximal(x - lambda * grad, lambda * alpha));
    end
end

