clc;clear;
%% PNA FNPGNN FXPNA TVNOA
%%
%Motion blur
P = fspecial('motion', 5, 70);
%Gaussian blur
%P = fspecial('gaussian', [5 5], 1);
n=256;
%%
X_image = imread('1.jpg');%1.jpg\2.jpg
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
tspan = [0 2];
%%
x01 = zeros(n);
x02 = zeros(n);
x03 = zeros(n);
tic
%R
[t1,x1]=ode23(@(t, x) matrixODE1(t, x,P,q_red,n),tspan, x01(:));
x10 = reshape(x1(end,:), [n, n]);
%G
[t2,x2]=ode23(@(t, x) matrixODE1(t, x,P,q_green,n),tspan, x02(:));
x20 = reshape(x2(end,:), [n, n]);
%B
[t3,x3]=ode23(@(t, x) matrixODE1(t, x,P,q_blue,n),tspan, x03(:));
x30 = reshape(x3(end,:), [n, n]);
time1 = toc
% 
hc_image1 = cat(3, x10, x20, x30);
%%
%FNPGNN
Gx01 = zeros(n);
Gx02 = zeros(n);
Gx03 = zeros(n);
tic
%R
[Gt1,Gx1]=ode23(@(t, x) matrixODE2(t, x,P,q_red,n),tspan, Gx01(:));
Gx10 = reshape(Gx1(end,:), [n, n]);
%G
[Gt2,Gx2]=ode23(@(t, x) matrixODE2(t, x,P,q_green,n),tspan, Gx02(:));
Gx20 = reshape(Gx2(end,:), [n, n]);
%B
[Gt3,Gx3]=ode23(@(t, x) matrixODE2(t, x,P,q_blue,n),tspan, Gx03(:));
Gx30 = reshape(Gx3(end,:), [n, n]);
time2 = toc
% 
hc_image2 = cat(3, Gx10, Gx20, Gx30);
%%
%FXPNA
Px01 = zeros(n);
Px02 = zeros(n);
Px03 = zeros(n);
%R
tic
[Pt1,Px1]=ode23(@(t, x) matrixODE3(t, x,P,q_red,n),tspan, Px01(:));
Px10 = reshape(Px1(end,:), [n, n]);
%G
[Pt2,Px2]=ode23(@(t, x) matrixODE3(t, x,P,q_green,n),tspan, Px02(:));
Px20 = reshape(Px2(end,:), [n, n]);
%B
[Pt3,Px3]=ode23(@(t, x) matrixODE3(t, x,P,q_blue,n),tspan, Px03(:));
Px30 = reshape(Px3(end,:), [n, n]);
time3 = toc
%
hc_image3 = cat(3, Px10, Px20, Px30);
%%
%TVNOA
Fx01 = zeros(n);
Fx02 = zeros(n);
Fx03 = zeros(n);
tic
%R
[Ft1,Fx1]=ode23(@(t, x) matrixODE4(t, x,P,q_red,n),tspan, Fx01(:));
Fx10 = reshape(Fx1(end,:), [n, n]);
%G
[Ft2,Fx2]=ode23(@(t, x) matrixODE4(t, x,P,q_green,n),tspan, Fx02(:));
Fx20 = reshape(Fx2(end,:), [n, n]);
%B
[Ft3,Fx3]=ode23(@(t, x) matrixODE4(t, x,P,q_blue,n),tspan, Fx03(:));
Fx30 = reshape(Fx3(end,:), [n, n]);
time4= toc
%
hc_image4 = cat(3, Fx10, Fx20, Fx30);
%%
figure;
imshow(hc_image1);
truesize([M N]/5);
%%
figure;
imshow(hc_image2);
truesize([M N]/5);
%%
figure;
imshow(hc_image3);
truesize([M N]/5);
%%
figure;
imshow(hc_image4);
truesize([M N]/5);
%%
PSNR1=psnr(X_image, hc_image1)
PSNR2=psnr(X_image, hc_image2)
PSNR3=psnr(X_image, hc_image3)
PSNR4=psnr(X_image, hc_image4)
SSIM1=ssim(X_image, hc_image1)
SSIM2=ssim(X_image, hc_image2)
SSIM3=ssim(X_image, hc_image3)
SSIM4=ssim(X_image, hc_image4)
%%
function dXdt = matrixODE1(t, X,P,q, n)
alpha = 0.0001; 
lambda = 0.8;
A_da=40;
proximal = @(X, da) soft_thresholding(X, da);
x = reshape(X, [n, n]);
grad = imfilter(imfilter(x, P, 'conv', 'circular')-q,rot90(P, 2), 'conv', 'circular');
dxdt =-(A_da)*(x-proximal(x - lambda * grad, lambda * alpha));
dXdt = dxdt(:);
end
function dXdt = matrixODE2(t, X,P,q, n)
alpha = 0.0001;
lambda = 0.8; 
A_da = 10;
R_da1 = 0.5;
proximal = @(X, da) soft_thresholding(X, da);
x = reshape(X, [n, n]);
grad = imfilter(imfilter(x, P, 'conv', 'circular')-q,rot90(P, 2), 'conv', 'circular');
dxdt =-(A_da/(norm(x-proximal(x - lambda * grad, lambda * alpha)))^(1-R_da1))*(x-proximal(x - lambda * grad, lambda * alpha)) ;
dXdt = dxdt(:);
end
function dXdt = matrixODE3(t, X,P,q, n)
alpha = 0.0001;
lambda = 0.8; 
A_da = 10;
B_da = 10; 
R_da1 = 0.5;
R_da2 = 1.5;
proximal = @(X, da) soft_thresholding(X, da);
x = reshape(X, [n, n]);
grad = imfilter(imfilter(x, P, 'conv', 'circular')-q,rot90(P, 2), 'conv', 'circular');
dxdt =-(A_da/(norm(x-proximal(x - lambda * grad, lambda * alpha)))^(1-R_da1)+B_da/norm(x-proximal(x - lambda * grad, lambda * alpha))^(1-R_da2))*(x-proximal(x - lambda * grad, lambda * alpha)) ;
dXdt = dxdt(:);
end
function dXdt = matrixODE4(t, X,P,q, n)
alpha = 0.0001; 
lambda = 0.8;
A_da = 10;
B_da = 10;
R_da1 = 0.5;
R_da2 = 1.5;
w=0.1;
proximal = @(X, da) soft_thresholding(X, da);
x = reshape(X, [n, n]);
grad = imfilter(imfilter(x, P, 'conv', 'circular')-q,rot90(P, 2), 'conv', 'circular');
dxdt =-((A_da/(w+t))/(norm(x-proximal(x - lambda * grad, lambda * alpha)))^(1-R_da1)+(B_da/(w+t))/norm(x-proximal(x - lambda * grad, lambda * alpha))^(1-R_da2))*(x-proximal(x - lambda * grad, lambda * alpha)) ;
dXdt = dxdt(:);
end