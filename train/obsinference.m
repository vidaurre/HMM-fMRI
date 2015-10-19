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

rmeanH = hmm.train.meanH(end:-1:1);

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
    t0fMRI = sum(data.T(1:tr-1)); t1fMRI = sum(data.T(1:tr));
    t0Gamma = sum(T(1:tr-1)) - scutoff*(tr-1); t1Gamma = sum(T(1:tr)) - scutoff*tr;
  
    if ~hmm.train.factorX
        
        X.S{tr} = zeros(Ttr*ndim);
        [Q,R] = buildQR(hmm,tr);
          
        % regularisation given by the state
        C1 = zeros(Ttr*ndim); % covariance matrix
        for k=1:K
            if strcmp(hmm.train.covtype,'diag')
                S = diag(hmm.state(k).Omega.rate ./ hmm.state(k).Omega.shape);
            else
                S = hmm.state(k).Omega.rate ./ hmm.state(k).Omega.shape;
            end
            for t=1:Ttr
                ind = (1:ndim) + ndim*(t-1);
                C1(ind,ind) = C1(ind,ind) + Gamma(t0Gamma+t,k) * S;
            end
        end
        
        % Design matrix
        A = zeros(data.T(tr)*ndim,Ttr*ndim); %,T(tr)*ndim) ?    
        for t=1:data.T(tr)
            for n=1:ndim
                ind1 = n+ndim*(t-1);
                ind2 = ( (t-1)*ndim+n ) : ndim : ( (t-2+L)*ndim+n );
                A(ind1,ind2) = R;
            end
        end
        
        % variance due to uncertainty of the HRF coefficients
        M2 = zeros(Ttr*ndim);  
        for j1 = 1:Ttr
            l1 = max(1,j1-Ttr+L) : min(L,j1);
            for d = 0:L-1
                j2 = j1+d;
                if j2>Ttr, break; end
                l2 = l1 + d; valid = (l2<=L);
                for n=1:ndim
                    ind1 = ndim*(j1-1) + n;
                    ind2 = ndim*(j2-1) + n;
                    M2(ind1,ind2) = sum(diag(Q(n,l1(valid),l2(valid))));
                    M2(ind2,ind1) = M(ind1,ind2);
                end
            end
        end
        
        % Dependent variable (frmi signal)
        I1 = hmm.train.I1(t0fMRI+1:t1fMRI,:);
        b = data.Y(t0fMRI+1:t1fMRI,:); 
        meanH = hmm.train.meanH;
        for t=sum(T(1:tr-1))+1:sum(T(1:tr-1)) + cutoff(1)
            these_l = hmm.train.I1(t,:) > 0;
            these_y = hmm.train.I1(t,these_l);
            for n=1:ndim
                b(these_y) = b(these_y) - X.mu(t,n) * meanH(these_l);
            end
        end
        b = b'; b = b(:);

        S1 = A' * A;
        
        
        
        
        m = reshape(data.Y(t0fMRI+1:t0fMRI,:)',t1fMRI-t0fMRI,1);
        
        % Variance
        if strcmp(hmm.train.covtype,'diag')
            
            
            for t=t0+1:t1,
                these_l = hmm.train.I1(t,:) > 0;
                X.S(t,:) = 1 ./ ( X.S(t,:) + sigma .* sum(Q(:,these_l),2)' );
            end
        else
            
            
            for t=t0+1:t1,
                these_l = hmm.train.I1(t,:) > 0;
                X.S(t,:,:) = inv(permute(X.S(t,:,:),[2 3 1]) + diag( sigma' .* sum(Q(:,these_l),2)) );
            end
        end
        
    else % factozing X across time
        
        Ttr = T(tr)-scutoff;
        
        % Auxiliar variables Q, R and C
        [Q,R,C] = buildQRC(hmm,tr);
        
        
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
                c(this_l,:) = c(this_l,:) + sum( C(:,:,this_l)' .* X.mu(these_x,:));
            end
            m = R(:,these_l)' .* m - c;
            if length(these_y)==1, sm = m;
            else sm = sum(m);
            end
            X.mu(t,:) = permute(X.S(t,:,:),[2 3 1]) * (mu(t-t0,:) + sigma .* sm)';
        end
        
    end
    
end









