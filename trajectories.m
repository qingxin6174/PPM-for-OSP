% plot saddle point trajectories
clear;
t = 1:1800;
z1 = @(t)log(1+t);
z2 = @(t)log(log(exp(1)+t));
F{1} = z2(t).*exp(1i*z1(t));
F{2} = z2(t).*exp(1i*(t*pi+z2(t)));
F{3} = z2(t).*exp(1i*(t*2*pi/3+z2(t)));

for Env = 1:3
    DATA = reshape(F{Env},Env,length(t)/Env).';
    idx = reshape(1:1800,Env,length(t)/Env).';
    figure,plot3(real(DATA(:)),imag(DATA(:)),idx(:),'.');grid on
    writecell({'x','y','t'},strcat('Env',num2str(Env),'.csv'));
    writematrix([real(DATA(:)),imag(DATA(:)),idx(:)],strcat('Env',num2str(Env),'.csv'),'WriteMode','append')
end

