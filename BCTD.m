function [model] = BCTD(Y, varargin)
% Rank-Revealing Functional Bayesian Tensor Decomposition
%   [model] = BCTD(Y, varargin) performs tensor decomposition
%   on the input tensor Y. It is commonly used for tensor completion (when O is provided).
%
%   Inputs:
%       Y        - The observed tensor (can be incomplete).
%       varargin - Optional parameter-value pairs:
%           'init'     - Initialization method ('ml' for maximum likelihood, 'rand' for random). Default: 'ml'.
%           'maxRank'  - Maximum rank R for the decomposition. Default: max(size(Y)).
%           'dimRed'   - Flag for dynamic rank reduction (1 to enable, 0 to disable). Default: 0.
%           'maxiters' - Maximum number of iterations. Default: 100.
%           'verbose'  - Display progress (1 to enable, 0 to disable). Default: 1.
%           'O'        - Observation indicator tensor (0 for unobserved, 1 for observed). Default: zeros(size(Y)).
%           'L'        - Cell array containing kernel matrices for each mode.
%                        Default: cell array of identity matrices.
%           'Xori'     - Ground truth tensor. Default: zeros(size(Y)).
%
%   Outputs:
%       model    - Structure containing the results:
%           .U       - Cell array of factor matrices {U_1, U_2, ..., U_K}.
%           .X       - The reconstructed tensor.
%           .USigma  - Cell array of covariance matrices for the factor components.
%           .lambdas - Precision hyperparameters for the factors.
%           .SNR     - Estimated Signal-to-Noise Ratio (dB).
%           .iternum - Number of iterations performed.
%           .TrueRank- Estimated rank of the final decomposition.
%           .tau     - Estimated noise precision.
%           .ranklist- List of estimated ranks across iterations.
%
%   Reference: When Bayesian Tensor Completion Meets Multioutput Gaussian Processes: 
%              Functional Universality and Rank Learning. IEEE TSP.

%% Read Parameters and Initialization

dimY = size(Y);
K = ndims(Y);

% Default L (Identity matrices for no coupling)
defaultL = cell(K, 1);
for k = 1:K
    defaultL{k} = eye(dimY(k));
end

ip = inputParser;
ip.addParameter('init', 'ml', @(x) (ismember(x,{'ml','rand'})));
ip.addParameter('maxRank', max(dimY), @isscalar);
ip.addParameter('dimRed', 0, @isscalar);
ip.addParameter('maxiters', 100, @isscalar);
ip.addParameter('verbose', 1, @isscalar);
ip.addParameter('noise', 'off', @(x)ismember(x,{'on','off'}));

% Input Tensors/Matrices
ip.addParameter('O', zeros(dimY),@(x)(isnumeric(x)||islogical(x))); % 0/1 indicator tensor
ip.addParameter('L', defaultL, @iscell); % A cell that contains kernel matrix at different modes
ip.addParameter('Xori', zeros(dimY),@(x)(isnumeric(x)||islogical(x))); % Ground truth (for evaluation)

ip.parse(varargin{:});

init  = ip.Results.init;
R   = ip.Results.maxRank;
maxiters  = ip.Results.maxiters;
verbose  = ip.Results.verbose;
DIMRED   = ip.Results.dimRed;
L = ip.Results.L;
Xori = ip.Results.Xori;
O = ip.Results.O;

Y = tensor(Y);
Nobs = sum(O(:));
ranklist = zeros(maxiters, 1);

% Noise precision hyperparameters (Gamma prior)
a0 = 1e-6;
b0 = 1e-6;
tau = 1; % Noise precision initialization

% Factor precision hyperparameters (Gamma prior for lambda)
c0 = 1e-6;
d0 = 1e-6;
lambdas = ones(R,1); % Factor precision initialization

% Data scaling
dscale = 1;
Y = Y./dscale;

% Initialization of factor matrices U and their covariance USigma
Yinit = ~O.*rand(dimY) + Y; % Fill missing entries with random values for ML init

U = cell(K,1);
USigma = cell(K,1);

for k = 1:K
    % Initialize component covariance matrices to identity
    USigma{k} = cell(R,1);
    for r=1:R
        USigma{k}{r} = eye(dimY(k)); % Covariance matrix for each component's factor vector
    end

    switch init
        case 'ml'    % Maximum Likelihood based on truncated SVD
            [A, S, ~] = svd(double(tenmat(Yinit,k)), 'econ');
            if R <= size(A,2)
                U{k} = A(:,1:R)*(S(1:R,1:R)).^(0.5);
            else
                % Pad with random if R is larger than matrix rank
                U{k} = [A*(S.^(0.5)) randn(dimY(k), R-size(A,2))];
            end
        case 'rand'   % Random initialization
            U{k} = randn(dimY(k),R);
    end
end

X = double(ktensor(U));

fprintf('\n----------Model Learning Begin----------\n')
for it = 1:maxiters
    
    %% Update Factor Matrices U_k (column by column)
    for k = 1:K % For each mode k
        for r = 1:R % For each component r 
            KRUrUr = 1;
            indexk = [1:k-1,k+1:K];
            for s=indexk(end:-1:1)
                KRUrUr = kr(KRUrUr,U{s}(:,r).*U{s}(:,r) + diag(USigma{s}{r}));%注意考虑方差
            end
            % updat variance
            USigma{k}{r} = inv( tau*diag(double(tenmat(O,k))*KRUrUr) + lambdas(r)*inv(L{k}) );

            OProdUkr = 1;
            for s=1:K
                OProdUkr = outprod(OProdUkr,U{s}(:,r));
            end

            % compute E(U_{\r}^{T} U_{\r})
            ErUTU = double(ktensor(U)) - OProdUkr;

            KRUr = 1;
            for s=indexk(end:-1:1)
                KRUr = kr(KRUr,U{s}(:,r));
            end

            % update mean U_r^k
            U{k}(:,r) = tau*USigma{k}{r}*double(tens2mat(O.*(double(Y) - ErUTU),k))*KRUr;
        end
    end
    
    % Reconstruct the current mean tensor X
    X = double(ktensor(U));
    
    %% Update Hyperparameter Lambda
    ck = (0.5*sum(dimY) + c0)*ones(R,1);
    dk = zeros(R,1);
    for r = 1:R
        urTLur = 0;
        for k=1:K
            urTLur = urTLur + 0.5.*( U{k}(:,r)'*inv(L{k})*U{k}(:,r) + trace(inv(L{k})*USigma{k}{r}) );
        end
        dk(r) = d0 + urTLur;
    end
    lambdas = ck./dk;
    
    %% Update Noise Precision Tau
    ak = a0 + 0.5* Nobs;

    % update from BTD-FC
    err = Y(:)'*Y(:) - 2*Y(:)'*X(:);
    for l=1:R
        term1 = 0;
        for p=1:R
            if p ~= l
                KRmlmp = 1;
                for s=K:-1:2
                    KRmlmp = kr(KRmlmp,U{s}(:,l).*U{s}(:,p));
                end
                term1 = term1 + (U{1}(:,l).*U{1}(:,p))'*double(tenmat(O,1))*KRmlmp;
            end
        end

        KRmlmp2 = 1;
        for s=K:-1:2
            KRmlmp2 = kr(KRmlmp2, U{s}(:,l).*U{s}(:,l) + diag(USigma{s}{l}));
        end
        term2 = (U{1}(:,l).*U{1}(:,l)+diag(USigma{1}{l}))' * double(tenmat(O,1)) * KRmlmp2;
        err = err + term1 + term2;
    end
    

    bk = b0 + 0.5*err;
    tau = ak/bk;
    
    %% Prune out unnecessary components (Dynamic Rank Reduction)
    
    % Compute the 'power' of each component r: sum_k ||U{k}(:,r)||^2
    Uall = cell2mat(U);
    comPower = diag(Uall' * Uall);
    
    % Threshold for component power (tolerance based on machine epsilon and total power)
    comTol = sum(dimY)*eps(norm(Uall,'fro'));
    rankest = sum(comPower > comTol);
    
    if max(rankest) == 0
        if verbose, disp('\\\======= Rank becomes 0! Converged===========\\\'); end
        break;
    end
    
    if DIMRED == 1 && it >= 2
        if R ~= rankest
            indices = comPower > comTol;
            for k=1:K
                U{k} = U{k}(:,indices);
                % USigma is a cell array of covariance matrices. Select corresponding cells.
                USigma{k} = USigma{k}(indices); 
            end
            lambdas = lambdas(indices);
            R = rankest;
        end
    end
    
    %% Display Progress
    if verbose
        fprintf('Iter. %d: Error = %g, Rank R = %d \n', it, frob(double(Xori-X)).^2, rankest);
    end
    
    % Record the rank
    ranklist(it) = rankest;
    
end

%% Prepare the Results
% Estimate SNR
SNR = 10*log10(var(X(:))*tau);

% Scale back the reconstructed tensor
X = ktensor(U)*dscale;
X = arrange(X); % Sort factors by power
X = double(X);

%% Output Model Structure
model.U = U;
model.X = X;
model.USigma = USigma;
model.lambdas = lambdas;
model.SNR = SNR;
model.iternum = it;
model.TrueRank = rankest;
model.dscale = dscale;
model.tau = tau;
model.ranklist = ranklist;

end