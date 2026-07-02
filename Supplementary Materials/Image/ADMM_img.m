clc; clear;
n=256;
%%
%Motion blur
P = fspecial('motion', 5, 70);
%Gaussian blur
%P = fspecial('gaussian', [5 5], 1);
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
%% ADMM
alpha = 0.0001;     
rho = 1;        
max_iter = 20000;   
tol1 = 1e-7; 
tol2 = 1e-3;          

tic;
P_otf = psf2otf(P, [n, n]);        
denom = abs(P_otf).^2 + rho;        

PTq_red = real(ifft2(conj(P_otf) .* fft2(q_red)));
PTq_green = real(ifft2(conj(P_otf) .* fft2(q_green)));
PTq_blue = real(ifft2(conj(P_otf) .* fft2(q_blue)));

x_red = q_red;   x_green = q_green;   x_blue = q_blue;
z_red = zeros(n); z_green = zeros(n); z_blue = zeros(n);
u_red = zeros(n); u_green = zeros(n); u_blue = zeros(n);

for iter = 1:max_iter

    rhs_red = PTq_red + rho * (z_red - u_red);
    rhs_green = PTq_green + rho * (z_green - u_green);
    rhs_blue = PTq_blue + rho * (z_blue - u_blue);
    
    x_red = real(ifft2( fft2(rhs_red) ./ denom ));
    x_green = real(ifft2( fft2(rhs_green) ./ denom ));
    x_blue = real(ifft2( fft2(rhs_blue) ./ denom ));
    
    v_red = x_red + u_red;
    v_green = x_green + u_green;
    v_blue = x_blue + u_blue;
    
    thr = alpha / rho;
    z_red_o=z_red;z_green_o=z_green;z_blue_o=z_blue;
    z_red = sign(v_red) .* max(abs(v_red) - thr, 0);
    z_green = sign(v_green) .* max(abs(v_green) - thr, 0);
    z_blue = sign(v_blue) .* max(abs(v_blue) - thr, 0);
    
    u_red = u_red + (x_red - z_red);
    u_green = u_green + (x_green - z_green);
    u_blue = u_blue + (x_blue - z_blue);

    r_primal = norm(x_red - z_red, 'fro') + norm(x_green - z_green, 'fro') + norm(x_blue - z_blue, 'fro');
    r_dual = (norm(z_red - z_red_o, 'fro') + norm(z_green - z_green_o, 'fro') + norm(z_blue - z_blue_o, 'fro'));

    if (r_primal < tol1) && (r_dual < tol2)
        fprintf('ADMM converged at iter %d\n', iter);
        break;
    end
    if mod(iter, 100) == 0
        fprintf('Iter %d, Primal=%.2e, Dual=%.2e\n', iter, r_primal, r_dual);
    end
end
time = toc
T_x10 = x_red;
T_x20 = x_green;
T_x30 = x_blue;

hc_image1 = cat(3, T_x10, T_x20, T_x30);
%%
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
