function [hmm] = HRFupdate (data,T,hmm,X)

ndim = size(X.mu,2);
N = length(T);
p = size(hmm.train.H,1);

for tr=1:N
    
    y = gety(data,T,tr,hmm,X); % frmi signal - where the start and end needs to be ajusted
    
    for n=1:ndim
        
        [G,V] = buildGV(data.T,T,hmm,X,n,tr); GG = G' *  G;
        
        % B - mixing coefficients (p X ndim)
        regterm = (hmm.HRF(tr).alpha.shape(n) /  hmm.HRF(tr).alpha.rate(n)) * hmm.HRF(tr).prior.B.iS;
        precision = hmm.HRF(tr).sigma.shape(n) / hmm.HRF(tr).sigma.rate(n);
        hmm.HRF(tr).B.S(:,:,n) = inv(precision*(GG + V) + regterm);
        hmm.HRF(tr).B.mu(:,n) = precision * hmm.HRF(tr).B.S(:,:,n) * ...
            (G' * y(:,n) + regterm * hmm.HRF(tr).prior.B.mu);
        
        % alpha - scaling of the mixing coefficients prior
        muB = hmm.HRF(tr).B.mu(:,n) - hmm.HRF(tr).prior.B.mu;
        S = hmm.HRF(tr).B.S(:,:,n) .* hmm.HRF(tr).prior.B.iS;
        hmm.HRF(tr).alpha.rate(n) = hmm.HRF(tr).prior.alpha.rate(n) + ...
            0.5 * (muB' * hmm.HRF(tr).prior.B.iS * muB + sum(S(:))); % - ...
            %2 * hmm.HRF(tr).B.mu(:,n)' * hmm.HRF(tr).prior.B.iS * hmm.HRF(tr).prior.B.mu );
        hmm.HRF(tr).alpha.shape(n) = hmm.HRF(tr).prior.alpha.shape(n) + 0.5 * p;
        
        % sigma
        hmm.HRF(tr).sigma.rate(n) = hmm.HRF(tr).prior.sigma.rate(n) + ...
            0.5 * sum((y(:,n) - G * hmm.HRF(tr).B.mu(:,n)).^2) + ...
            0.5 * sum(sum((V .* hmm.HRF(tr).B.S(:,:,n))));
        hmm.HRF(tr).sigma.shape(n) = hmm.HRF(tr).prior.sigma.shape(n) + 0.5 * data.T(tr);
        
    end
    
end

end