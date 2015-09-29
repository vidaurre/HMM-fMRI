function X = obsinit2(data,hmm,Xinit)

meanH = hmm.train.meanH;

L = length(meanH);
T = data.T + L - 1;
ndim = size(data.Y,2);
regularisation = 0.1;

X.mu = zeros(sum(T),ndim);
if strcmp(hmm.train.covtype,'diag')
    X.S = zeros(sum(T),ndim);
else
    X.S = zeros(sum(T),ndim,ndim);
end
[~,maxH] = max(meanH);
for tr=1:length(T)
    t0 = sum(T(1:tr-1)); t1 = sum(T(1:tr));
    t0fMRI = sum(data.T(1:tr-1)); t1fMRI = sum(data.T(1:tr));
    %X.mu(t0+1:t1,:) = [zeros(L-maxH,ndim); data.Y(t0fMRI+1:t1fMRI,:); zeros(maxH-1,ndim)];
    if nargin<3
        X.mu(t0+1:t1,:) = [zeros(L-1,ndim); data.Y(t0fMRI+1:t1fMRI,:)]; % only if HzfMRI == Hz\
    else
        X.mu(t0+1:t1,:) = Xinit(t0+1:t1,:);
    end
end

for it=1:hmm.train.init_iterations
    
    for tr=1:length(T)
        
        t0 = sum(T(1:tr-1)); t1 = sum(T(1:tr));
        
        % Auxiliar variables Q, R
        
        Q = repmat(meanH.^2,ndim,1);
        R = repmat(meanH,ndim,1);
        
        % Variance
        for t=t0+1:t1,
            these_l = hmm.train.I1(t,:) > 0;
            if strcmp(hmm.train.covtype,'diag')
                X.S(t,:) = 1 ./ ( sum(Q(:,these_l),2)' + regularisation*ones(1,ndim) );
            else
                X.S(t,:,:) = diag(1./ (sum(Q(:,these_l),2) + regularisation*ones(ndim,1) ) );
            end
        end
        
        % Mean
        for t=t0+1:t1,
            these_l = hmm.train.I1(t,:) > 0;
            these_y = hmm.train.I1(t,these_l);
            m = data.Y(these_y,:);
            for tt = these_y
                this_l = find(these_y == tt);
                these_x = hmm.train.I2(tt,:); these_x(these_x==t) = [];
                these_no_l = 1:L; these_no_l(hmm.train.I2(tt,:)==t) = [];
                m(this_l,:) = m(this_l,:) - sum(R(:,these_no_l)' .* X.mu(these_x,:));  %sum(R(:,these_no_l),2)';
            end
            m = R(:,these_l)' .* m;
            if length(these_y)==1, sm = m;
            else sm = sum(m);
            end
            X.mu(t,:) = permute(X.S(t,:,:),[2 3 1]) * sm';
        end
        
    end
    
end

end
