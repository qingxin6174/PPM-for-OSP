function [aDG,aNE] = optoppm(T,idx,delay)
xx = 2*rand-1; yy = 2*rand-1;
u = zeros(1,T); v = zeros(1,T); P = zeros(2,T);
delta = zeros(2,T); dg = zeros(1,T); ne = zeros(1,T); %reg = zeros(2,T);
a = zeros(1,T); b = zeros(1,T); % (a,b) is the saddle point of h(t)
pu = 8; pv = 8; e = 0.1; Dx = 8; Dy = 8; L1 = 12; L2 = 12;
for t=1:T
    % update learning rates
    eta = L1*(Dx+pu)/(e+sum(delta(1:t-1)));
    gamma = L2*(Dy+pv)/(e+sum(delta(2:t-1)));
    % observe h(t)
    if t>delay
        at = a(t-delay); bt = b(t-delay);
    else
        at = 2*rand-1; bt = 2*rand-1;
    end
    h = @(x,y) (x-at)^2/2-(y-bt)^2/2+(x-at)*(y-bt);
    [x,y] = saddlesolver(at,bt,xx,yy,eta,gamma);
    % output (x,y), observe f and compeditor
    [f,u(t),v(t),a(t),b(t)] = payoff(x,y,t,idx);
    % (a(t),b(t)) is the saddle point
    dg(t) = f(x,v(t))-f(u(t),y); % duality gap
    ne(t) = f(x,y)-f(a(t),b(t)); % NE-regret
    % reg(1,t) = f(x,y)-f(u(t),y); % regret 1
    % reg(2,t) = f(x,v(t))-f(x,y); % regret 2
    if t>1
        % obtain path-length
        P(:,t) = P(:,t-1)+[norm(u(t)-u(t-1));norm(v(t)-v(t-1))];
    end
    if P(1,t)>pu
        pu = 2*pu; % doubling trick for player 1
    end
    if P(2,t)>pv
        pv = 2*pv; % doubling trick for player 2
    end
    % update auxiliary parameter
    [xx,yy] = optsolver(a(t),b(t),x,y,xx,yy,eta,gamma);
    delta(1,t) = f(x,y)-h(x,y)+h(xx,y)-f(xx,y)-(x-xx)^2/2/eta;
    delta(2,t) = f(x,yy)-h(x,yy)+h(x,y)-f(x,y)-(y-yy)^2/2/gamma;
end
aDG = cumsum(dg)./(1:T); % average duality gap over time
aNE = abs(cumsum(ne))./(1:T); % average NE-regret over time
% aREG = cumsum(reg,2)./[1:T;1:T]; % average regret over time