function [vpath]=hmmdecode(T,hmm,X)
%
% Viterbi and single-state decoding for hmm
% The algorithm is run for the whole data set, including those whose class
% was fixed. This means that the assignment for those can be different.
%
% INPUT
% T         length of latent series
% hmm       hmm data structure
% X         latent series
%
% OUTPUT
% vpath(i).q_star    maximum likelihood state sequence
%
% Author: Diego Vidaurre, OHBA, University of Oxford

tiny=exp(-700);
N = length(T);
K=hmm.K;
P=hmm.P;
Pi=hmm.Pi;
cutoff = hmm.train.cutoff; scutoff = sum(abs(cutoff));

for tr=1:N
    
    Ttr = T(tr)-scutoff;
    q_star = ones(Ttr,1);
    
    alpha=zeros(Ttr,K);
    beta=zeros(Ttr,K);
    
    % Initialise Viterbi bits
    delta=zeros(Ttr,K);
    psi=zeros(Ttr,K);

    t0 = sum(T(1:tr-1));  
    Xin.mu = X.mu(t0+1+cutoff(1):t0+T(tr)+cutoff(2),:);
    Xin.S = cell(1);
    Xin.S{1} = X.S{tr};

    B = obslike(hmm,Xin);
    B(B<realmin) = realmin;
    
    scale=zeros(Ttr,1);
    % Scaling for delta
    dscale=zeros(Ttr,1);
    
    alpha(1,:)=Pi(:)'.*B(1,:);
    scale(1)=sum(alpha(1,:));
    alpha(1,:)=alpha(1,:)/(scale(1)+tiny);
    
    delta(1,:) = alpha(1,:);    % Eq. 32(a) Rabiner (1989)
    % Eq. 32(b) Psi already zero
    for i=2:Ttr
        alpha(i,:)=(alpha(i-1,:)*P).*B(i,:);
        scale(i)=sum(alpha(i,:));
        alpha(i,:)=alpha(i,:)/(scale(i)+tiny);
        
        for k=1:K,
            v=delta(i-1,:).*P(:,k)';
            mv=max(v);
            delta(i,k)=mv*B(i,k);  % Eq 33a Rabiner (1989)
            if length(find(v==mv)) > 1
                % no unique maximum - so pick one at random
                tmp1=find(v==mv);
                tmp2=rand(length(tmp1),1);
                [~,tmp4]=max(tmp2);
                psi(i,k)=tmp4;
            else
                psi(i,k)=find(v==mv);  % ARGMAX; Eq 33b Rabiner (1989)
            end
        end;
        
        % SCALING FOR DELTA ????
        dscale(i)=sum(delta(i,:));
        delta(i,:)=delta(i,:)/(dscale(i)+tiny);
    end;
    
    % Get beta values for single state decoding
    beta(Ttr,:)=ones(1,K)/scale(Ttr);
    for i=Ttr-1:-1:1
        beta(i,:)=(beta(i+1,:).*B(i+1,:))*(P')/scale(i);
    end;
    
    xi=zeros(Ttr-1,K*K);
    for i=1:Ttr-1
        t=P.*( alpha(i,:)' * (beta(i+1,:).*B(i+1,:)));
        xi(i,:)=t(:)'/sum(t(:));
    end;
    
    % Backtracking for Viterbi decoding
    id = find(delta(Ttr,:)==max(delta(Ttr,:)));% Eq 34b Rabiner;
    q_star(Ttr) = id(1);
    for i=Ttr-1:-1:1,
        q_star(i) = psi(i+1,q_star(i+1));
    end
    
    vpath(tr).q_star = q_star;
    
end

