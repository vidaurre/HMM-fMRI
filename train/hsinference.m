function [Gamma,Gammasum,Xi,LL]=hsinference(data,T,hmm,X)
%
% inference engine for HMMs.
%
% INPUT
%
% data      Observations - a struct with X (time series) and C (classes)
% T         Number of time points for each time series
% hmm       hmm data structure
%
% OUTPUT
%
% Gamma     Probability of hidden state given the data
% Gammasum  sum of Gamma over t
% Xi        joint Prob. of child and parent states given the data
%
% Author: Diego Vidaurre, OHBA, University of Oxford


N = length(T);
K=hmm.K;
cutoff = hmm.train.cutoff; scutoff = sum(abs(cutoff));

Gamma=[]; LL = [];
Gammasum=zeros(N,K);
Xi=[];

for in=1:N
    t0 = sum(T(1:in-1));  
    Xin.mu = X.mu(t0+1+cutoff(1):t0+T(in)+cutoff(2),:);
    if strcmp(hmm.train.covtype,'diag')
        Xin.S = X.S(t0+1+cutoff(1):t0+T(in)+cutoff(2),:);
    else
        Xin.S = X.S(t0+1+cutoff(1):t0+T(in)+cutoff(2),:,:);
    end
    C = data.C(t0+1+cutoff(1):t0+T(in)+cutoff(2),:);
    % we jump over the fixed parts of the chain
    t = 1; Tin = T(in)-scutoff;
    xi = []; gamma = []; gammasum = zeros(1,K); ll = [];
    
    while t<=Tin
        if isnan(C(t,1)), no_c = find(~isnan(C(t:end,1)));
        else no_c = find(isnan(C(t:end,1)));
        end
        if t>1
            if isempty(no_c), slice = (t-1):Tin;  
            else slice = (t-1):(no_c(1)+t-2);  
            end;
        else
            if isempty(no_c), slice = t:Tin;  
            else slice = (t):(no_c(1)+t-2);  
            end;
        end
        if isnan(C(t,1))
            x.mu = Xin.mu(slice,:); 
            if strcmp(hmm.train.covtype,'diag'), x.S = Xin.S(slice,:);
            else x.S = Xin.S(slice,:,:); 
            end
            [gammat,xit,Bt]=nodecluster(x,hmm); 
        else
            gammat = zeros(length(slice),K);
            if t==1, gammat(1,:) = C(slice(1),:); end
            xit = zeros(length(slice)-1, K^2);
            for i=2:length(slice)
                gammat(i,:) = C(slice(i),:);
                xitr = gammat(i-1,:)' * gammat(i,:) ;
                xit(i-1,:) = xitr(:)';
            end
        end
        if t>1,
            gammat = gammat(2:end,:);
        end
        xi = [xi; xit];
        gamma = [gamma; gammat];
        gammasum = gammasum + sum(gammat);
        if nargout==4, ll = [ll; log(sum(Bt(1:end,:) .* gammat,2)) ]; end
        if isempty(no_c), break;
        else t = no_c(1)+t-1;
        end;
    end
    Gamma = [Gamma; gamma];
    Gammasum(in,:) = gammasum;
    if nargout==4, LL = [LL;ll]; end
    Xi = cat(1,Xi,reshape(xi,T(in)-1-scutoff,K,K));
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Gamma,Xi,B]=nodecluster(X,hmm)
% inference using normal forward backward propagation
% there's a mistake here when used with C - it should not use Pi unless it
% is exactly at the first time point of the series

T = size(X.mu,1);
P=hmm.P;
K=size(P,2);
Pi=hmm.Pi;

B = obslike(hmm,X);
B(B<realmin) = realmin;

scale=zeros(T,1);
alpha=zeros(T,K);
beta=zeros(T,K);

alpha(1,:)=Pi.*B(1,:);
scale(1)=sum(alpha(1,:));
alpha(1,:)=alpha(1,:)/scale(1);
for i=2:T
    alpha(i,:)=(alpha(i-1,:)*P).*B(i,:);
    scale(i)=sum(alpha(i,:));		% P(X_i | X_1 ... X_{i-1})
    alpha(i,:)=alpha(i,:)/scale(i);
end;

beta(T,:)=ones(1,K)/scale(T);
for i=T-1:-1:1
    beta(i,:)=(beta(i+1,:).*B(i+1,:))*(P')/scale(i);
end;
Gamma=(alpha.*beta);
Gamma=Gamma(1:T,:);
Gamma=rdiv(Gamma,rsum(Gamma));

Xi=zeros(T-1,K*K);
for i=1:T-1
    t=P.*( alpha(i,:)' * (beta(i+1,:).*B(i+1,:)));
    Xi(i,:)=t(:)'/sum(t(:));
end

