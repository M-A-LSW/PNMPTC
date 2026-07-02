clc;clear;
%%
%Motion blur
P = fspecial('motion', 5, 70);
%Gaussian blur
%P = fspecial('gaussian', [5 5], 1);
%%
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
blurred_image2 = cat(3, q_red, q_green, q_blue);
tspan = [0 1];
%%
tic;
T_x01 = zeros(n);
T_x02 =  zeros(n);
T_x03 =  zeros(n);
%R
[T_t1,T_x1]=ode23(@(t, x) matrixODE1(t, x,P,q_red,n),tspan, T_x01(:));
T_x10 = reshape(T_x1(end,:), [n, n]);
%G
[T_t2,T_x2]=ode23(@(t, x) matrixODE1(t, x,P,q_green,n),tspan, T_x02(:));
T_x20 = reshape(T_x2(end,:), [n, n]);
%B
[T_t3,T_x3]=ode23(@(t, x) matrixODE1(t, x,P,q_blue,n),tspan, T_x03(:));
T_x30 = reshape(T_x3(end,:), [n, n]);
time=toc
%
hc_image1 = cat(3, T_x10, T_x20, T_x30);
%%
for i = 1:length(T_t1)
    mwc1(i)=norm(X_image(:,:,1)-reshape(T_x1(i,:), [n, n]));
end
for i = 1:length(T_t2)
    mwc2(i)=norm(X_image(:,:,2)-reshape(T_x2(i,:), [n, n]));
end
for i = 1:length(T_t3)
    mwc3(i)=norm(X_image(:,:,3)-reshape(T_x3(i,:), [n, n]));
end
%
SSIM=ssim(X_image, hc_image1)
PSNR=psnr(X_image, hc_image1)
%%
figure;
imshow(X_image);
truesize([M, N]/5);
figure;
imshow(im2double(imresize(blurred_image2, [M, N]/5)));
truesize([M, N]/5);
figure;
imshow(im2double(imresize(hc_image1, [M, N]/5)));
truesize([M, N]/5);
%%
function dXdt = matrixODE1(t, X,P,q, n)
alpha = 0.0001; 
Tp = 1;
lambda = 0.8; 
r = 1;
proximal = @(X, da) soft_thresholding(X, da);
x = reshape(X, [n, n]);
grad = imfilter(imfilter(x, P, 'conv', 'circular')-q,rot90(P, 2), 'conv', 'circular');
u = @(t) r/(Tp-t)^1.2;
    if t <Tp-1e-12
        dxdt = -(u(t))* (x-proximal(x - lambda * grad, lambda * alpha));
    else
        dxdt = -r/(1e12)^1.2 * (x-proximal(x - lambda * grad, lambda * alpha));
    end
dXdt = dxdt(:);
end