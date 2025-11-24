%% demo for RR-FBTC on SSF data
clc;clear;close all;
seed = 1;
randn('state',seed); rand('state',seed); %#ok<RAND>
%% load SSF data and create observation
data = load('data.mat');%load SSF data
ssf = single(data.data);
x_true = ssf;
[N_x, N_y, N_z] = size(ssf);
I_total = prod([N_x, N_y, N_z]);

% Add noise
SNR_dB = 1e1; %% pre-defined SNR
if SNR_dB < 100
    signal_power = mean(ssf(:).^2);
    SNR = 10^(SNR_dB / 10);  
    noise_power = signal_power / SNR;
    sigma = sqrt(noise_power); 
else
    sigma = 0;
end
noisy_data = ssf + sigma * randn(N_x, N_y, N_z, 'single');

% Normalize
ssp_max = max(noisy_data(:));
ssp_min = min(noisy_data(:));
x = (noisy_data - ssp_min) / (ssp_max - ssp_min);

% Observation
nmod = 3;
total = N_x * N_y * N_z;
p = 0.3; % sampling ratio
N = floor(p * numel(x));
Omega = randperm(numel(x)); Omega = Omega(1:N);
ob = zeros(size(x));
ob(round(Omega)) = 1;
Y = x .* ob;

%% RR-FBTC
range = 10;%
mag = 1;%
func = @(h)mag*(1+sqrt(5)*h/range+5*h.^2/(3*range.^2))*exp(-sqrt(5)*h./(range));%matern 2.5

dimY = size(Y);
K = ndims(Y);
Karray = cell(K,1);
for d=1:K
    KMd = zeros(dimY(d),dimY(d));
    for i=1:dimY(d)
        for j=1:dimY(d)
            KMd(i,j)=func(abs(i-j));
        end
    end
    Karray{d} = KMd + eye(dimY(d))*(1e-6);
end

[model] = BCTD(Y,'O', ob, 'init', 'rand', 'maxRank', round(min(dimY)), 'dimRed', 1, 'Xori', x_true,...
                'maxiters', 100, 'verbose', 1, 'L', Karray);
X_FBTC = model.X;


a = abs((X_FBTC*(ssp_max - ssp_min)+ssp_min -x_true));
MAE_FBTC = sum(a(:))/(prod(dimY));
RMSE_FBTC = norm(double(tenmat(X_FBTC*(ssp_max - ssp_min)+ssp_min-x_true,1)),'fro')/sqrt(prod(dimY));


%% plot
figure
figureUnits = 'centimeters';
figureWidth = 15;
figureHeight = 15;
set(gcf, 'Units', figureUnits, 'Position', [8 12 figureWidth figureHeight]); 

num_methods = 3;
dep = [1 10 20];
for i=1:3   
    subplot(3,num_methods,num_methods*(i-1)+1)
        pcolor( squeeze( x_true(:,:,dep(i)) ) )
            shading interp;
            colormap('jet');
            axis image
    title('ground-truth');

    subplot(3,num_methods,num_methods*(i-1)+2)
    a = Y(:,:,dep(i));
    a(a==0)=nan;
    [m,n]=size(a);
    [X,Y_sample] = meshgrid(0:n-1,0:m-1);
    imagesc(X(1,:),Y_sample(:,1),flipud(a));
    axis image
    colormap('jet');
    title('measurements')

        subplot(3,num_methods,num_methods*(i-1)+3);
    pcolor( squeeze( X_FBTC(:,:,dep(i)))  )
            shading interp;
            colormap('jet');
            axis image
    title(' RR-FBTC'); 

end
