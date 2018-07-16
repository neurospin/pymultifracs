
function data_out = fractint(data_in,alpha);

taille = length(data_in);

% data_in=data_in-mean(data_in);
% data_in=data_in-linspace(data_in(1),data_in(end),taille);

%% Fourier Transform
X=fft(data_in);
% frequency
k=-taille/2:taille/2-1;

%% Filtre
H = alpha - 1;
Kmod=abs(k);

listnull=find(Kmod==0);
Kmod(listnull)=min(Kmod(setdiff(1:length(Kmod(:)),listnull)))*ones(1,length(listnull));
Hfilt=1./(Kmod.^(1+H));


% figure(1)
% clf
% plot(log((k(end/2+2:end))),log(Hfilt(end/2+2:end)),'r-')
% hold on
% fx=abs(fft(data_in));
% plot(log((k(end/2+2:end))),log(fx(2:end/2)),'ko')

%% Filtrage et FFT inverse
Y=Hfilt.*fftshift(X);
Y=ifftshift(Y);
data_out=ifft(Y);


% function y=fractint(x,alp)
% %% function y=fractint(x,alp)
% %
% % FFT based fractional integration / derivative of periodic function x
% % the mean value is removed
% %
% % HW, TLS, 08/2013
% 
% N=length(x); x=x-mean(x);
% P=floor((N+1)/2);    % zero frequency index of MATLAB DFT
% k=(0:N-1)/N-(P-1)/N; % frequencies
% kalpha=(1i*k).^-alp; kalpha(P)=0; % Weyl fractional integration
% tfy=fftshift(fft(x)).*kalpha/2/pi;
% y=real(ifft(ifftshift(tfy)));


