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

if strcmp(hmm.train.covtype,'diag')
    X.S = zeros(sum(T),ndim);
else
    X.S = zeros(sum(T),ndim,ndim);
end

Tminus = T - scutoff;

for tr=1:length(T)
    
    sigma = hmm.HRF(tr).sigma.shape ./ hmm.HRF(tr).sigma.rate;
    
    t0 = sum(T(1:tr-1)) + cutoff(1); t1 = sum(T(1:tr)) + cutoff(2);
    t0Gamma = sum(Tminus(1:tr-1));  t1Gamma = sum(Tminus(1:tr)); 
    
    % Auxiliar variables Q, R and C
    [Q,R,C] = buildQRC(hmm,tr);
    
    % Variance
    if strcmp(hmm.train.covtype,'diag')
        Gammaplus = repmat(Gamma(t0Gamma+1:t1Gamma,:),[1,1,ndim]);  Gammaplus = permute(Gammaplus,[1 3 2]);
        for k=1:K
            S = repmat((hmm.state(k).Omega.shape ./ hmm.state(k).Omega.rate),Tminus(tr),1);
            X.S(t0+1:t1,:) =  X.S(t0+1:t1,:) + Gammaplus(:,:,k) .* S;
        end
        for t=t0+1:t1,
            these_l = hmm.train.I1(t,:) > 0;
            X.S(t,:) = 1 ./ ( X.S(t,:) + sigma .* sum(Q(:,these_l),2)' );
        end
    else
        Gammaplus = repmat(Gamma(t0Gamma+1:t1Gamma,:),[1,1,ndim,ndim]);  Gammaplus = permute(Gammaplus,[1 3 4 2]);
        for k=1:K
            S = repmat(hmm.state(k).Omega.shape .* hmm.state(k).Omega.irate, [1,1,Tminus(tr)]);
            X.S(t0+1:t1,:,:) = X.S(t0+1:t1,:,:) + Gammaplus(:,:,:,k) .* permute(S,[3 1 2]);
        end
        for t=t0+1:t1,
            these_l = hmm.train.I1(t,:) > 0;
            X.S(t,:,:) = inv(permute(X.S(t,:,:),[2 3 1]) + diag( sigma' .* sum(Q(:,these_l),2)) );
        end
    end
    
    % Mean
    Gammaplus = repmat(Gamma(t0Gamma+1:t1Gamma,:),[1,1,ndim]);  Gammaplus = permute(Gammaplus,[1 3 2]);
    mu = zeros(Tminus(tr),ndim);
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









