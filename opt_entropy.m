function o = opt_entropy(E,alpha)
A = E(:); n = length(A); s = sum(A);
W = 1:n;
c = 0; C = 0;
while ~isempty(W)
    a = median(A(W));
    L = W(A(W)<a); M = W(A(W)==a); H = W(A(W)>a);
    if a*(1-(c+length(L))*alpha/n)/(s-(C+sum(A(L)))) < alpha/n
        c = c+length(L)+length(M);
        C = C+sum(A(L))+sum(A(M));
        if isempty(H)
            a = min(A(A>a));
        end
        W = H;
    else
        W = L;
    end
end
o = A*(1-c*alpha/n)/(s-C); o(A<a) = alpha/n; o = reshape(o,size(E)); 