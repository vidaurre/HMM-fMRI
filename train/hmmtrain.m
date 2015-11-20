function [hmm,Gamma,Xi,X,fehist,actstates]=hmmtrain(data,T,hmm,X,Gamma)
%
% Train Hidden Markov Model using using Variational Framework
%
% INPUTS:
%
% data          observations - a struct with X (time series) and C (classes)
% T             Number of time points for each latent time series
% hmm           hmm structure with options specified in hmm.train
% Gamma         Initial state courses
% X             Initial latent signal
%
% OUTPUTS
% hmm           estimated HMMMAR model
% Gamma         estimated p(state | data)
% Xi            joint probability of past and future states conditioned on data
% X             Latent signal
% fehist        historic of the free energies across iterations
% knocked       states knocked out by the Bayesian inference
%
% hmm.Pi          - intial state probability
% hmm.P           - state transition matrix
% hmm.state(k).$$ - whatever parameters there are in the observation model
%
% Author: Diego Vidaurre, OHBA, University of Oxford

K = hmm.train.K;
ndim = size(data.Y,2);

fehist=[];
actstates = ones(1,K);
%cyc_to_go = 0;
% tmp = Inf;

% load /tmp/X_orig.mat; load /tmp/hmm_orig.mat;
% if 0
% for k=1:K,
%     if strcmp(hmm.train.covtype,'full')
%     hmm.state(k).Omega.shape = hmm_orig.state(k).Omega.shape;
%     hmm.state(k).Omega.rate = hmm_orig.state(k).Omega.rate;
%     hmm.state(k).Omega.irate = hmm_orig.state(k).Omega.irate;
%     else
%     hmm.state(k).Omega.shape = hmm_orig.state(k).Omega.shape;
%     hmm.state(k).Omega.rate = diag(hmm_orig.state(k).Omega.rate)';
%     end
%     if strcmp(hmm.train.covtype,'full')
%      hmm.state(k).Mean.mu = hmm_orig.state(k).Mean.mu; % full
%      hmm.state(k).Mean.S = hmm_orig.state(k).Mean.S;
%     else
%      hmm.state(k).Mean.mu = hmm_orig.state(k).Mean.mu; % diag
%      hmm.state(k).Mean.S = diag(hmm_orig.state(k).Mean.S)';
%     end
% %     hmm.state(k).Mean.mu = ones(1,2);
% %     hmm.state(k).Mean.S = ones(1,2);
% %     hmm.state(k).Omega.rate = ones(1,2);
% %     hmm.state(k).Omega.shape = 1;
% end
% for tr=1:10
%     hmm.HRF(tr).B.mu = hmm_orig.HRF(tr).B.mu;
%     hmm.HRF(tr).B.S = hmm_orig.HRF(tr).B.S;
%     hmm.HRF(tr).sigma.shape = hmm_orig.HRF(tr).sigma.shape;
%     hmm.HRF(tr).sigma.rate = hmm_orig.HRF(tr).sigma.rate;
%     %hmm.HRF(tr).sigma.shape = ones(1,2);
%     %hmm.HRF(tr).sigma.rate = ones(1,2);
%     hmm.HRF(tr).alpha.shape = ones(1,2);
%     hmm.HRF(tr).alpha.rate = ones(1,2);
% end
% end
% if 0
%     X.mu = X_orig;
%     for tr=1:length(T)
%         X.S{tr} = .1 * eye(size(X.S{tr},2));
%     end
% end
% load /tmp/Gamma_orig.mat;  GammaAux = [];
% for tr=1:length(T)
%     t0 = sum(T(1:tr-1));
%     gammainit = Gamma_orig(t0+1+hmm.train.cutoff(1):t0+T(tr)+hmm.train.cutoff(2),:);
%     GammaAux = [GammaAux; gammainit];
% end

for cycle=1:hmm.train.cyc
    
    if size(Gamma,2)==1 && hmm.train.stopwhenK1, break; end
    
    %%% X and HRF part
    for subcycle=1:hmm.train.subcyc(1)
        
        %[hmm_orig.state(1).Omega.rate/hmm_orig.state(1).Omega.shape hmm_orig.state(2).Omega.rate/hmm_orig.state(2).Omega.shape ...
        % hmm_orig.state(3).Omega.rate/hmm_orig.state(3).Omega.shape;
        %  hmm.state(1).Omega.rate/hmm.state(1).Omega.shape hmm.state(2).Omega.rate/hmm.state(2).Omega.shape ...
        % hmm.state(3).Omega.rate/hmm.state(3).Omega.shape  ]
        %          figure(2);
        %          subplot(1,2,1);plot([X.mu(1:327*3,1) X_orig(1:327*3,1)]);
        %          hold on; colormap('pink');plot(3*Gamma_orig(1:327*3,:)-1.5,'LineWidth',4);colormap('jet'); hold off
        %          subplot(1,2,2);plot([X.mu(1:327*3,2) X_orig(1:327*3,2)]);
        %          hold on; colormap('pink');plot(3*Gamma_orig(1:327*3,:)-1.5,'LineWidth',4);colormap('jet'); hold off
        %          figure(3);subplot(2,1,1);plot(Gamma(1:1000,:));subplot(2,1,2);plot(GammaAux(1:1000,:));ylim([-0.1 1.1])
        
        % Latent signal
        %          if cycle==1
        X = obsinference(data,T,hmm,Gamma,X);
%         if cycle>1
%             tmp0 = tmp;
%             tmp = sum(evalfreeenergy(data,T,hmm,Gamma,Xi,X));
%             if tmp0<tmp, fprintf('After X: WRONG %g \n',tmp-tmp0);end
%         end
        %          end
        
        % HRF model
        if cycle<=hmm.train.subcycHRF(1) || mod(cycle,hmm.train.subcycHRF(2))==0
            %             if cycle==1
            hmm = HRFupdate(data,T,hmm,X);
%             if cycle>1
%                 tmp0 = tmp;
%                 tmp = sum(evalfreeenergy(data,T,hmm,Gamma,Xi,X));
%                 if tmp0<tmp, fprintf('After HRF: WRONG %g \n',tmp-tmp0); end
%             end
            %             end
            %load /tmp/hmmtrue.mat
            %HRF = reconstructHRF(hmm);
            %HRFtrue = reconstructHRF(hmmtrue);
            %clf;plot(HRF');hold on; plot(HRFtrue',':'); hold off
        end
        
    end
    
    %%% Gamma, observational model and transition matrix
    for subcycle=1:hmm.train.subcyc(2)
        
        % Gaussian observation model
        %if cycle==1
        hmm = obsupdate(hmm,Gamma,X,T);
%         if cycle>1
%             tmp0 = tmp;
%             tmp = sum(evalfreeenergy(data,T,hmm,Gamma,Xi,X));
%             if tmp0<tmp, fprintf('After hmm: WRONG %g \n',tmp-tmp0); end
%         end
        %end
        
        % State time courses
        if size(Gamma,2)>1 % otherwise all states have already collapsed
            [Gamma,~,Xi]=hsinference(data,T,hmm,X);
%             if cycle>1
%                 tmp0 = tmp;
%                 tmp = sum(evalfreeenergy(data,T,hmm,Gamma,Xi,X));
%                 if tmp0<tmp, fprintf('After gamma: WRONG %g \n',tmp-tmp0); end
%             end
            if any(isnan(Gamma(:))),
                warning('Some Gamma values have been assigned NaN')
                keyboard;
            end
            if any(mean(Gamma)>0.95) && hmm.train.stopwhenK1,
                warning('All states have collapsed in one')
            end
            % any state to remove?
            as1 = find(actstates==1);
            [as,hmm,Gamma,Xi] = getactivestates(hmm,Gamma,Xi);
            if any(as==0),
                %warning('I am dropping one (or more) states')
                cyc_to_go = hmm.train.cycstogoafterevent;
            end
            actstates(as1(as==0)) = 0;
        end

        % transition matrices and initial state
        %if cycle==1
        hmm = hsupdate(hmm,T,Xi,Gamma);
%         if cycle>1
%             tmp0 = tmp;
%             tmp = sum(evalfreeenergy(data,T,hmm,Gamma,Xi,X));
%             if tmp0<tmp, fprintf('After Theta: WRONG , %g \n',tmp-tmp0); end
%         end
        %end
        
    end
    
    %%% Free energy computation
    fe = sum(evalfreeenergy(data,T,hmm,Gamma,Xi,X));
    fehist = [fehist; fe];
    if cycle==1
        fprintf('cycle %i free energy = %g \n',cycle,fehist(end));
    else
        chgFrEn = (fehist(end)-fehist(end-1))/fehist(end-1);
        fprintf('cycle %i free energy = %g (relative change %g) \n',cycle,fehist(end),chgFrEn);
        if (abs(chgFrEn) < hmm.train.tol), break; end
    end
    
end

if hmm.train.verbose
    fprintf('Model: %d kernels, %d dimension(s), %d data samples \n',K,ndim,sum(T));
end

end

