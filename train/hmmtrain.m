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

N = length(T); K = hmm.train.K;
ndim = size(data.Y,2);

fehist=[];
actstates = ones(1,K);
%cyc_to_go = 0;

leave = 0; % borrar

for cycle=1:hmm.train.cyc
    
    if leave, break; end % borrar
    
    %%% X and HRF part
    for subcycle=1:hmm.train.subcyc(1)
        
        % Latent signal
        X = obsinference(data,T,hmm,Gamma,X);
        
        % HRF model
        if cycle<=hmm.train.subcycHRF(1) || mod(cycle,hmm.train.subcycHRF(2))==0
            hmm = HRFupdate(data,T,hmm,X);
        end
        
    end
       
    %%% Gamma, observational model and transition matrix
    for subcycle=1:hmm.train.subcyc(2)
        
        % Gaussian observation model
        hmm = obsupdate(hmm,Gamma,X);
        
        % State time courses
        [Gamma,~,Xi]=hsinference(data,T,hmm,X);
        
        if any(isnan(Gamma(:))),
            warning('Some Gamma values have been assigned NaN')
            keyboard;
        end
        if any(mean(Gamma)>0.95) && hmm.train.stopwhenK1,    
            warning('All states have collapsed in one')
            leave = 1;
        end
        
        % any state to remove?
        as1 = find(actstates==1);
        [as,hmm,Gamma,Xi] = getactivestates(hmm,Gamma,Xi);
        if any(as==0), 
            warning('I am dropping a state')
            cyc_to_go = hmm.train.cycstogoafterevent; end
        actstates(as1(as==0)) = 0;
        
        % transition matrices and initial state
        if hmm.train.updateGamma,
           hmm = hsupdate(hmm,T,Xi,Gamma);
        end
        
    end
    
    %%% Free energy computation
     fehist = [fehist; sum(evalfreeenergy(data,T,hmm,Gamma,Xi,X))];
%     strwin = ''; if hmm.train.meancycstop>1, strwin = 'windowed'; end
%     if cycle>(hmm.train.meancycstop+1) && cyc_to_go==0
%         chgFrEn[ = mean( fehist(end:-1:(end-hmm.train.meancycstop+1)) - fehist(end-1:-1:(end-hmm.train.meancycstop)) )  ...
%             / (fehist(1) - fehist(end));
%         if hmm.train.verbose, fprintf('cycle %i free energy = %g, %s relative change = %g \n',cycle,fehist(end),strwin,chgFrEn); end
%         if (abs(chgFrEn) < hmm.train.tol), break; end
%     elseif hmm.train.verbose && cycle>1, fprintf('cycle %i free energy = %g \n',cycle,fehist(end));
%     end
%    if cyc_to_go>0, cyc_to_go = cyc_to_go - 1; end
    if cycle==1
        fprintf('cycle %i free energy = %g \n',cycle,fehist(end));
    else
        fprintf('cycle %i free energy = %g (change %g) \n',cycle,fehist(end),(fehist(end)-fehist(end-1))/fehist(end-1));
    end
        
end

if hmm.train.verbose
    fprintf('Model: %d kernels, %d dimension(s), %d data samples \n',K,ndim,sum(T));
end

return;

