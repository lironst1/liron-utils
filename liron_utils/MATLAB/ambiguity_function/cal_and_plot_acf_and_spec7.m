% cal_and_plot_acf_and_spec7.m - Written by Eli Mozeson and Nadav Levanon
% plot corrrelation function and spectrum of a signal defined by
% u(t), t, and F (maximal Dopper normalized by signal length)

% calcualte normalized ACF
acfun=20*log10(abs(xcorr(u))+eps);
acfun=acfun-max(acfun);
acfun1=acfun;
acfun=max(acfun,-70);
acfun=acfun(1:(length(acfun)+1)/2);
acfun=fliplr(acfun);
scalet=[0:length(acfun)-1]/(length(acfun)-1)*t(length(t));

% calcualte spectrum
fftlength=max(1024*8,length(u)*32);
spec=20*log10(max(abs(fft(u,fftlength)),eps));
spec=spec-max(spec);
spec=max(spec,-60);
spec=spec(1:fftlength/2);
scales=[0:fftlength/2-1]/(fftlength/2-1)*r/2;

% calcualte normalized PACF
zeru=zeros(size(u));
pacfun=20*log10(abs(xcorr([u u u],[zeru u zeru]))+eps);
clear zeru;
pacfun=pacfun-max(pacfun);
pacfun=max(pacfun,-70);
pacfun=pacfun(3*length(u):4*length(u)-1);

figure(acffig);
set(acffig,'Visible','on');
subplot(311);
plot(scalet,acfun,'k');
xlabel('{\it\tau}/\itt_b');
ylabel('Autocorrelation [dB]');
title(titlest);
axis([0 max(scalet) -70 0]);

subplot(312);
plot(scalet,pacfun,'k');
xlabel('{\it\tau}/\itt_b');
ylabel('Periodic Autocorrelation [dB]');
axis([0 max(scalet) -70 0]);

subplot(313);
plot(scales*length(u)/r,spec,'k');
xlabel('{\itf*Mt_b}');
ylabel('|{\itS}({\itf})|');
axis([0 F -60 0]);
