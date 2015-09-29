function [hmm] = HRFupdate (data,T,hmm,X)

ndim = size(X.mu,2); 
N = length(T);
p = size(hmm.train.H,1);

for tr=1:N

    t0 = sum(data.T(1:tr-1)); t1 = sum(data.T(1:tr));
    %hmm.HRF(tr).B.mu = zeros(p,ndim);
    %hmm.HRF(tr).B.S = zeros(p,p,ndim);
    
    for n=1:ndim

        [G,V] = buildGV(data.T(tr),hmm,X,n); GG = G' *  G;
        
        % B
         regterm = (hmm.HRF(tr).alpha.shape(n) /  hmm.HRF(tr).alpha.rate(n)) * hmm.HRF(tr).prior.B.iS;
         precision = hmm.HRF(tr).sigma.shape(n) / hmm.HRF(tr).sigma.rate(n);
         hmm.HRF(tr).B.S(:,:,n) = inv(precision*(GG + V) + regterm);
         if hmm.train.zeroBmean==1
             hmm.HRF(tr).B.mu(:,n) = precision * hmm.HRF(tr).B.S(:,:,n) * G' * data.Y(t0+1:t1,n);
         else
             hmm.HRF(tr).B.mu(:,n) = precision * hmm.HRF(tr).B.S(:,:,n) * (G' * data.Y(t0+1:t1,n) + regterm * hmm.HRF(tr).prior.B.mu);
         end
        
        % alpha
        %muB = hmm.HRF(tr).B.mu(:,n) + hmm.HRF(tr).prior.B.mu;
        %S = hmm.HRF(tr).B.S(:,:,n) .* hmm.HRF(tr).prior.B.iS; 
        %hmm.HRF(tr).alpha.rate(n) = hmm.HRF(tr).prior.alpha.rate(n) + ...
        %     0.5 * (muB' * hmm.HRF(tr).prior.B.iS * muB + sum(S(:)) - ...
        %     2 * hmm.HRF(tr).B.mu(:,n)' * hmm.HRF(tr).prior.B.iS * hmm.HRF(tr).prior.B.mu );
        %hmm.HRF(tr).alpha.shape(n) = hmm.HRF(tr).prior.alpha.shape(n) + 0.5 * p;      
        
        % sigma
        hmm.HRF(tr).sigma.rate(n) = hmm.HRF(tr).prior.sigma.rate(n) + ...
           0.5*sum((data.Y(t0+1:t1,n) - G * hmm.HRF(tr).B.mu(:,n)).^2) + ...
           0.5*trace(V * hmm.HRF(tr).B.S(:,:,n));
        hmm.HRF(tr).sigma.shape(n) = hmm.HRF(tr).prior.sigma.shape(n) + 0.5*data.T(tr);
    
    end
            
end

end