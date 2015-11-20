function [X,updateXindexes] = obsinit(data,hmm)
% updateXindexes refers to indexes of the time points in X that are
% going to be actualized during VB training. The time points at the
% beginning and the end of each trial are only estimated by obsinit and
% they are left intact for the rest of the process
%
% Regularization is given by the parameter hmm.train.lambda;
% if hmm.train.lambda < 0, then it is a Bayesian estimation, 
%           with prior shape/rate precision of the prior on X given by lambda
% if hmm.train.lambda == 0, then a plugin estimator is used
% if hmm.train.lambda > 0, a fixed lambda is used 

meanH = hmm.train.meanH; 
L = length(meanH);
rmeanH = meanH(L:-1:1); 
T = data.T + L - 1;
ndim = size(data.Y,2);
cutoff = hmm.train.cutoff; scutoff = sum(abs(cutoff));
updateXindexes = [];
lambda = hmm.train.lambda;

X.mu = zeros(sum(T),ndim);
X.S = cell(length(T),1);

for tr = 1:length(data.T)
    t0fMRI = sum(data.T(1:tr-1)); t1fMRI = sum(data.T(1:tr));
    t0 = sum(T(1:tr-1)); t1 = sum(T(1:tr));
    updateXindexes = [updateXindexes (t0+cutoff(1)+1):(t1+cutoff(2))];
    Ttr = T(tr)-scutoff;
    X.S{tr} = single(0.001 * eye(Ttr*ndim)); 
    A = zeros(data.T(tr),T(tr)); 
    p = size(A,2);
    for t=1:data.T(tr)
        ind = (1:L) + (t-1);
        A(t,ind) = rmeanH;
    end
    C = A' * A; 
    for n=1:ndim % estimation is done for each channel independently
        b = data.Y(t0fMRI+1:t1fMRI,n);
        if lambda == 0 
            X.mu(t0+1:t1,n) = (C + 0.1 * mean(diag(C)) * eye(p)) \ A' * b;    
        elseif lambda > 0
            X.mu(t0+1:t1,n) = (C + lambda * eye(p)) \ A' * b;   
        else
            X.mu(t0+1:t1,n) = bayesregr(A,b,-lambda);
        end
    end
end
    
end


function X = bayesregr(A,b,prior,cycles)

[N,p] = size(A);
if nargin<3, cycles = 0.1;end
if nargin<4, cycles = 10;end
sigma = 1; 
prec = prior;
X0 = zeros(p,1);
C = A' * A;
for cyc = 1:cycles
    X = (sigma * C + prec * eye(p)) \ (sigma * A' * b); 
    if sum(abs(X-X0))<1e-4; break; end
    X0 = X; 
    sigma = (N + 0.2) / ( sum( (A*X - b).^2 ) + 0.2 )  ;
    prec = (p + (prior*2))  / ( sum(X.^2) + (prior*2) ) ;
end

end
