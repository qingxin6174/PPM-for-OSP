function [aDG,aNE] = oppm(T,idx)
x = zeros(1,T); y = zeros(1,T); x(1) = 2*rand-1; y(1) = 2*rand-1;
u = zeros(1,T); v = zeros(1,T); P = zeros(1,T); f = cell(1,T);
Delta = zeros(1,T); delta = zeros(1,T); Sigma = zeros(1,T); 
dg = zeros(1,T); ne = zeros(1,T);
p = 8; e = 0.1; Dx = 8; Dy = 8; L1 = 12; L2 = 12;
for t=1:T
    % output (x(t),y(t)), observe f(t) and compeditor
    [f{t},u(t),v(t),a,b] = payoff(x(t),y(t),t,idx);
    % (a,b) is the saddle point
    dg(t) = f{t}(x(t),v(t))-f{t}(u(t),y(t)); % duality gap
    ne(t) = f{t}(x(t),y(t))-f{t}(a,b); % NE-regret
    if t>1
        % obtain path-length
        P(t) = P(t-1)+norm(u(t)-u(t-1))+norm(v(t)-v(t-1));
        % update auxiliary parameter
        delta(t-1) = f{t-1}(u(t),y(t))-f{t-1}(u(t-1),y(t-1)) ...
            +f{t-1}(x(t-1),v(t-1))-f{t-1}(x(t),v(t)) ...
            -(x(t)-x(t-1))^2/2/eta-(y(t)-y(t-1))^2/2/gamma;
        Sigma(t-1) = max(0,sum(delta(1:t-1)));
        Delta(t-1) = max(0,Sigma(t-1)-sum(max(Sigma(1:t-2))));
    end
    if P(t)>p
        p = 2*p; % doubling trick
    end
    eta = 2*L1*(Dx+Dy+p)/(e+sum(Delta(1:t-2)));
    gamma = L2/L1*eta; % update learning rates
    [x(t+1),y(t+1)] = saddlesolver(a,b,x(t),y(t),eta,gamma);
end
aDG = cumsum(dg)./(1:T); % average duality gap over time
aNE = abs(cumsum(ne))./(1:T); % average NE-regret over time