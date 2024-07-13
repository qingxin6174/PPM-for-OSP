function [f,u,v,a,b] = payoff(x,y,t,idx)
% [f,u,v,a,b] = payoff(x,y,t,idx)
% (x,y) is the strategy pair, f is the payoff function
% (u,v) is the worst-case compeditor, (a,b) is the saddle point
z1 = @(t)log(1+t);
z2 = @(t)log(log(exp(1)+t));
% Initialize 4 cases
saddlepoint{1} = @(s,t) z2(t)*exp(1i*z1(t));
saddlepoint{2} = @(s,t) z2(t)*exp(1i*(t*pi+z2(t)));
saddlepoint{3} = @(s,t) z2(t)*exp(1i*(t*2*pi/3+z2(t)));
saddlepoint{4} = @(s,t) sqrt(2)*exp(1i*(8*pi/9+angle(s)));
% obtain saddle point
sp = saddlepoint{idx}(x+1i*y,t);
a = real(sp); b = imag(sp);
f = @(x,y) (x-a).^2/2-(y-b).^2/2+(x-a).*(y-b);
u = a+b-y;
v = b-a+x;
end