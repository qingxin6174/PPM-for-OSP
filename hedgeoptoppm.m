function [aDG,aNE] = hedgeoptoppm(T,idx,delay)
xx = 2*rand-1; yy = 2*rand-1;
u = zeros(1,T); v = zeros(1,T); P = zeros(2,T);
delta = zeros(2,T); dg = zeros(1,T); ne = zeros(1,T);
a = zeros(1,T); b = zeros(1,T); % (a,b) is the saddle point of h(t)
pu = 8; pv = 8; e = 0.1; Dx = 8; Dy = 8; L1 = 12; L2 = 12;
d = length(delay); at = zeros(1,d); bt = zeros(1,d);
w = ones(d,1)/d; Loss = ones(d,1); sigma = zeros(1,T); Round = 32;
for t=1:T
    % update learning rates
    eta = L1*(Dx+pu)/(e+sum(delta(1:t-1)));
    gamma = L2*(Dy+pv)/(e+sum(delta(2:t-1)));
    % observe h(t)
    for i = 1:d
        if t > delay(i)
            at(i) = a(t-delay(i)); bt(i) = b(t-delay(i));
        else
            at(i) = 2*rand-1; bt(i) = 2*rand-1;
        end
    end
    h = @(x,y) x^2/2+x*y-y^2/2-at*w*(x+y)-bt*w*(x-y)+(at.^2/2+at.*bt-bt.^2/2)*w;
    [x,y] = saddlesolver(at*w,bt*w,xx,yy,eta,gamma);
    % output (x,y), observe f and compeditor
    [f,u(t),v(t),a(t),b(t)] = payoff(x,y,t,idx);
    % (a(t),b(t)) is the saddle point
    dg(t) = f(x,v(t))-f(u(t),y); % duality gap
    ne(t) = f(x,y)-f(a(t),b(t)); % NE-regret
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
    % obtain meta-loss L
    for i = 1:d
        h = @(x,y) (x-at(i))^2/2-(y-bt(i))^2/2+(x-at(i))*(y-bt(i));
        Loss(i) = max([abs(f(x,y)-h(x,y));abs(f(xx,y)-h(xx,y));abs(f(x,yy)-h(x,yy))]);
    end
    if t > Round
        Round = 2*Round;
    end
    theta = log(Round)/(e+sum(sigma));
    w_old = w;
    w = opt_entropy(w.*exp(-theta*Loss),d/Round);
    sigma(t) = Loss'*(w_old-w)-w'*log(w./w_old)/theta;
end
aDG = cumsum(dg)./(1:T); % average duality gap over time
aNE = abs(cumsum(ne))./(1:T); % average NE-regret over time