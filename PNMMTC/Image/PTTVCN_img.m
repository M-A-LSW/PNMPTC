clc;clear;
%%
P = fspecial('motion', 5, 70);
%Gaussian blur
%P = fspecial('gaussian', [5 5], 1);
n=256;
%%
X_image = imread('1.jpg');
[M,N,K]=size(X_image);
X_image = imresize(X_image, [n, n]);
X_image = im2double(X_image);
red_channel = X_image(:,:,1);
green_channel = X_image(:,:,2);
blue_channel = X_image(:,:,3);
%%
rng(4);
Ns=0.001*randn(n,n);
q_red = imfilter(red_channel, P, 'conv', 'circular')+Ns;
q_green = imfilter(green_channel, P, 'conv', 'circular')+Ns;
q_blue = imfilter(blue_channel, P, 'conv', 'circular')+Ns;
blurred_image = cat(3, q_red, q_green, q_blue);
%%
tspan = [0 1];
%%
Fx01 = zeros(n);
Fx02 = zeros(n);
Fx03 = zeros(n);
%R
tic;
[Ft1,Fx1]=ode23(@(t, x) matrixODE(t, x,P,q_red,n),tspan, Fx01(:));
Fx10 = reshape(Fx1(end,:), [n, n]);
%G
[Ft2,Fx2]=ode23(@(t, x) matrixODE(t, x,P,q_green,n),tspan, Fx02(:));
Fx20 = reshape(Fx2(end,:), [n, n]);
%B
[Ft3,Fx3]=ode23(@(t, x) matrixODE(t, x,P,q_blue,n),tspan, Fx03(:));
Fx30 = reshape(Fx3(end,:), [n, n]);
time = toc
%%
hc_image4 = cat(3, Fx10, Fx20, Fx30);
%%
figure;
imshow(hc_image4);
truesize([M N]/5);
%%
PSNR=psnr(X_image, hc_image4)
SSIM=ssim(X_image, hc_image4)
%%
function dXdt = matrixODE(t, X,P,q, n)
%%
alpha = 0.0001; 
lambda = 0.8; 
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
Pp =(Tp*(1/(c1*a1*(1-Ck1))+1/(c2*a2*(Ck2-1))))/log((w+Tp)/w);
proximal = @(X, da) soft_thresholding(X, da);
x = reshape(X, [n, n]);
grad = imfilter(imfilter(x, P, 'conv', 'circular')-q,rot90(P, 2), 'conv', 'circular');
dxdt =-(Pp/Tp)*((c1/(w+t))/(norm(x-proximal(x - lambda * grad, lambda * alpha)))^(1-R_da1)+(c2/(w+t))/norm(x-proximal(x - lambda * grad, lambda * alpha))^(1-R_da2))*(x-proximal(x - lambda * grad, lambda * alpha)) ;
dXdt = dxdt(:);
end