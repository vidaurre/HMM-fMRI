function X = obsinference(data,T,hmm,Gamma,X,tr)
%
% inference engine for HMMs.
%
% INPUT
%
% data      Observations - a struct with X (time series) and C (classes)
% T         Number of time points for each time series
% hmm       hmm data structure
% Gamma     Probability of hidden state given the data
%
% OUTPUT
%
% X             Latent signal
%
% Author: Diego Vidaurre, OHBA, University of Oxford
 
[~,L] = size(hmm.train.H);
ndim = size(data.Y,2);
K = size(Gamma,2);
cutoff = hmm.train.cutoff; scutoff = sum(abs(cutoff));
initializing = all(Gamma(:)==0);

if nargin<6
    segments = 1:length(T);
else
    segments = tr;
end
if ~isfield(X,'S'), X.S = cell(length(T),1); end

for tr=segments
    
    sigma = hmm.HRF(tr).sigma.shape ./ hmm.HRF(tr).sigma.rate;
    Ttr = T(tr)-scutoff;
    
    t0 = sum(T(1:tr-1)) + cutoff(1); t1 = sum(T(1:tr)) + cutoff(2);
    t0Gamma = sum(T(1:tr-1)) - scutoff*(tr-1); t1Gamma = sum(T(1:tr)) - scutoff*tr;
    
    [Q,R] = buildQR(hmm,tr);
    Sigma = repmat(single(sigma),1,data.T(tr));
    
    % regularisation given by the state (prior of the variance)
    M1 = single(zeros(Ttr*ndim)); % covariance matrix
    if ~initializing
        for k=1:K
            if strcmp(hmm.train.covtype,'diag')
                S = diag(hmm.state(k).Omega.rate ./ hmm.state(k).Omega.shape);
            else
                S = hmm.state(k).Omega.rate ./ hmm.state(k).Omega.shape;
            end
            for t=1:Ttr
                ind = (1:ndim) + ndim*(t-1);
                M1(ind,ind) = M1(ind,ind) + single(Gamma(t0Gamma+t,k) * S);
            end
        end
        M1 = single(inv(M1));
    end
    
    % Design matrix
    A = single(zeros(data.T(tr)*ndim,Ttr*ndim));
    for t=1:data.T(tr)
        if t<=cutoff(1)
            J0 = cutoff(1)-t+1; J1 = 0; % how many x_t to leave out
            tt1 = 1; tt2 = L-J0;
        elseif t>(data.T(tr)+cutoff(2))
            J0 = 0; J1 = t - (data.T(tr)+cutoff(2));
            tt1 =  t - cutoff(1); tt2 = Ttr;
        else
            J0 = 0; J1 = 0;
            tt1 = t - cutoff(1); tt2 = tt1 + L - 1;
        end
        for n=1:ndim
            ind1 = n+ndim*(t-1);
            ind2 = ( (tt1-1)*ndim+n ) : ndim : ( (tt2-1)*ndim+n );
            A(ind1,ind2(end:-1:1)) = single(R(n,1+J1:end-J0));  
        end
    end
    
    % variance due to uncertainty of the HRF coefficients
    M2 = single(zeros(Ttr*ndim));
    for j1 = (1+cutoff(1)):(Ttr+cutoff(1))
        l1 = max(1,L-j1+1): min(Ttr+scutoff-j1+1,L);
        for d = 0:L-1
            j2 = j1 + d;
            if j2>(Ttr+cutoff(1)), break; end
            l2 = l1 + d; valid = (l2<=L);
            for n=1:ndim
                ind1 = ndim*(j1-cutoff(1)-1) + n;  % no need to reverse anything?
                ind2 = ndim*(j2-cutoff(1)-1) + n;
                M2(ind1,ind2) = single(sigma(n) * sum(diag(Q(l1(valid),l2(valid),n))));
                M2(ind2,ind1) = M2(ind1,ind2);
            end
        end
    end
    
    % Dependent variable (frmi signal), where the start and end needs to be ajusted
    y = gety(data,T,tr,hmm,X); 
    y = y'; y = y(:);
    
    % Prior of the mean
    mu0 = zeros(Ttr,ndim);
    for k=1:K
        mu0 = mu0 + repmat(Gamma(t0Gamma+1:t1Gamma,k),1,ndim) .* repmat(hmm.state(k).Mean.mu,Ttr,1);
    end
    mu0 = mu0'; mu0 = mu0(:);
    
    Asigm = A' .* repmat(Sigma,size(A,2),1);  
    X.S{tr} = inv(Asigm * A + M1 + M2);
    mu = X.S{tr} * ( Asigm * y + M1 * mu0 );
    %if tr==9, keyboard; end
    X.mu(t0+1:t1,:) = reshape(mu,ndim,Ttr)';   
    
end
