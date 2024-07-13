function [xx,yy] = optsolver(a,b,x,y,x0,y0,eta,gamma)
xx = (a*eta + b*eta + x0 - eta*y)/(1 + eta);
yy = (-a*gamma + b*gamma + gamma*x + y0)/(1 + gamma);
if xx < -2.5
    xx=-2.5;
end
if xx > 2.5
    xx=2.5;
end
if yy < -2.5
    yy=-2.5;
end
if yy > 2.5
    yy=2.5;
end
end