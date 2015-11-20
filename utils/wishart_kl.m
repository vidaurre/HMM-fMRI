function [D] = wishart_kl (B_q,B_p,alpha_q,alpha_p,logdetB_q,logdetB_p)
% computes the divergence between two k-dimensional Wishart prop. densities
%                /
%      D(q||p) = | q(x)*log(q(x)/p(x)) dx
%               /

if nargin<4,
  error('Incorrect number of input arguments');
end;

if size(B_q)~=size(B_p),
  error('Distributions must have equal dimensions');
end;

K=size(B_p,1);

if nargin<5, logdetB_q = logdet(B_q,'chol'); end
if nargin<6, logdetB_p = logdet(B_p,'chol'); end


Lq = -logdetB_q + K * log(2);
Lp = -logdetB_p + K * log(2);

lZq = log(2) * (alpha_q*K/2)  - Lq * (-alpha_q/2) + K*(K-1)/4 * log(pi); 
lZp = log(2) * (alpha_p*K/2)  - Lp * (-alpha_p/2) + K*(K-1)/4 * log(pi); 

Lq = Lq + K * log(2);
Lp = Lp + K * log(2);

for k=1:K
    lZq = lZq + gammaln(alpha_q/2+0.5-0.5*k);
    lZp = lZp + gammaln(alpha_p/2+0.5-0.5*k);
    Lq = Lq + psi(alpha_q/2+0.5-0.5*k);
    Lp = Lp + psi(alpha_p/2+0.5-0.5*k);
end

D = (alpha_q/2-0.5-0.5*K)*Lq - (alpha_p/2-0.5-0.5*K)*Lp ...
    - alpha_q * K / 2 + alpha_q * trace(B_p*inv(B_q)) / 2 + lZp - lZq;

return;

