function X = obsinference(data,T,hmm,Gamma,X)
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

if hmm.train.factorX
    if strcmp(hmm.train.covtype,'diag')
        X.S = zeros(sum(T),ndim);
    else
        X.S = zeros(sum(T),ndim,ndim);
    end
else
    X.S = cell(length(T),1);
end


for tr=1:length(T)
    
    sigma = hmm.HRF(tr).sigma.shape ./ hmm.HRF(tr).sigma.rate;
    Ttr = T(tr)-scutoff;
    
    t0 = sum(T(1:tr-1)) + cutoff(1); t1 = sum(T(1:tr)) + cutoff(2);
    t0Gamma = sum(T(1:tr-1)) - scutoff*(tr-1); t1Gamma = sum(T(1:tr)) - scutoff*tr;
  
    if ~hmm.train.factorX
        
        X.S{tr} = zeros(Ttr*ndim);
        [Q,R] = buildQR(hmm,tr);
        Sigma = diag(repmat(sigma,1,data.T(tr)));
                  
        % regularisation given by the state (prior of the variance)
        Cov = zeros(Ttr*ndim); % covariance matrix
        for k=1:K
            if strcmp(hmm.train.covtype,'diag')
                S = diag(hmm.state(k).Omega.rate ./ hmm.state(k).Omega.shape);
            else
                S = hmm.state(k).Omega.rate ./ hmm.state(k).Omega.shape;
            end
            for t=1:Ttr
                ind = (1:ndim) + ndim*(t-1);
                Cov(ind,ind) = Cov(ind,ind) + Gamma(t0Gamma+t,k) * S;
            end
        end
        if initializing
            M1 = zeros(Ttr*ndim);
        else
            M1 = inv(Cov);
        end
        
        % Design matrix
        A = zeros(data.T(tr)*ndim,Ttr*ndim);
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
                A(ind1,flip(ind2)) = R(n,1+J1:end-J0); % HERE
            end
        end
        
        % variance due to uncertainty of the HRF coefficients
        M2 = zeros(Ttr*ndim);   
        for j1 = (1+cutoff(1)):(Ttr+cutoff(1))
            l1 = max(1,L-j1+1): min(Ttr+scutoff-j1+1,L);
            for d = 0:L-1
                j2 = j1 + d;
                if j2>(Ttr+cutoff(1)), break; end
                l2 = l1 + d; valid = (l2<=L);  
                for n=1:ndim
                    ind1 = ndim*(j1-cutoff(1)-1) + n;  % no need to reverse anything? 
                    ind2 = ndim*(j2-cutoff(1)-1) + n;  
                    M2(ind1,ind2) = sum(diag(Q(l1(valid),l2(valid),n)));
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
        
        X.S{tr} = inv(A' * Sigma * A + M1 + M2);
        mu = X.S{tr} * ( A' * Sigma * y + M1 * mu0);
        X.mu(t0+1:t1,:) = reshape(mu,ndim,Ttr)';  % high var occurs at the end - too short cutoff(2)? - Check other trials
        
    else % factozing X across time
        
        % Auxiliar variables Q, R and C
        [Q,R,Cov] = buildQRC(hmm,tr);
        
        % Variance
        if strcmp(hmm.train.covtype,'diag')
            Gammaplus = repmat(Gamma(t0Gamma+1:t1Gamma,:),[1,1,ndim]);  Gammaplus = permute(Gammaplus,[1 3 2]);
            for k=1:K
                S = repmat((hmm.state(k).Omega.shape ./ hmm.state(k).Omega.rate),Ttr,1);
                X.S(t0+1:t1,:) =  X.S(t0+1:t1,:) + Gammaplus(:,:,k) .* S;
            end
            for t=t0+1:t1,
                these_l = hmm.train.I1(t,:) > 0;
                X.S(t,:) = 1 ./ ( X.S(t,:) + sigma .* sum(Q(:,these_l),2)' );
            end
        else
            Gammaplus = repmat(Gamma(t0Gamma+1:t1Gamma,:),[1,1,ndim,ndim]);  Gammaplus = permute(Gammaplus,[1 3 4 2]);
            for k=1:K
                S = repmat(hmm.state(k).Omega.shape .* hmm.state(k).Omega.irate, [1,1,Ttr]);
                X.S(t0+1:t1,:,:) = X.S(t0+1:t1,:,:) + Gammaplus(:,:,:,k) .* permute(S,[3 1 2]);
            end
            for t=t0+1:t1,
                these_l = hmm.train.I1(t,:) > 0;
                X.S(t,:,:) = inv(permute(X.S(t,:,:),[2 3 1]) + diag( sigma' .* sum(Q(:,these_l),2)) );
            end
        end
        
        % Mean
        Gammaplus = repmat(Gamma(t0Gamma+1:t1Gamma,:),[1,1,ndim]);  Gammaplus = permute(Gammaplus,[1 3 2]);
        mu = zeros(Ttr,ndim);
        for k=1:K
            if strcmp(hmm.train.covtype,'diag')
                S = hmm.state(k).Omega.shape ./ hmm.state(k).Omega.rate;
                mu = mu + Gammaplus(:,:,k) .*  repmat(S .* hmm.state(k).Mean.mu, size(Gammaplus,1),1  );
            else
                S = hmm.state(k).Omega.shape .* hmm.state(k).Omega.irate;
                mu = mu + Gammaplus(:,:,k) .*  repmat( (S * hmm.state(k).Mean.mu')', size(Gammaplus,1),1  );
            end
        end
        for t=t0+1:t1
            these_l = hmm.train.I1(t,:) > 0;
            these_y = hmm.train.I1(t,these_l);
            m = data.Y(these_y,:);
            c = zeros(length(these_y),ndim);
            for tt = these_y
                this_l = find(these_y == tt);
                these_x = hmm.train.I2(tt,:); these_x(these_x==t) = [];
                these_no_l = 1:L; these_no_l(hmm.train.I2(tt,:)==t) = [];
                m(this_l,:) = m(this_l,:) - sum(R(:,these_no_l)' .* X.mu(these_x,:));
                c(this_l,:) = c(this_l,:) + sum( Cov(:,:,this_l)' .* X.mu(these_x,:));
            end
            m = R(:,these_l)' .* m - c;
            if length(these_y)==1, sm = m;
            else sm = sum(m);
            end
            X.mu(t,:) = permute(X.S(t,:,:),[2 3 1]) * (sigma .* mu(t-t0,:) + sigma .* sm)';
        end
        
    end
    
end
