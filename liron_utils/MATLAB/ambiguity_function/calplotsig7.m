% calplotsig7.m - Written by Eli Mozeson and Nadav Levanon
% used by ambfn7 for calculation and plot of the signal when the signal is defined
% by u_amp, u_phase and u_freq (single carrier signal)

% output variables include u(t) amd t on which u is defined

tb=1;
  
if (acode==1)*(pcode==1),
   u_basic=u_amp.*exp(j*u_phase*pi);
elseif (acode==1)*(pcode==0),
   u_basic=u_amp;
elseif (acode==0)*(pcode==1),
   u_basic=exp(j*u_phase*pi);
else
   u_basic=ones(size(u_amp));
end
  
m_basic=length(u_basic);

if r==1
    dt=tb;
    m=m_basic;
    uamp=abs(u_basic);
    phas=uamp*0;
    phas=angle(u_basic);
    if fcode==1
       phas=phas+2*pi*dt*cumsum(f_basic);
    end
    uexp=exp(j*phas);
    u=uamp.*uexp;
else                               % i.e., several samples within a bit
    dt=tb/r;                       % interval between samples
    ud=diag(u_basic);
    ao=ones(r,m_basic);
    m=m_basic*r;
    u_basic=reshape(ao*ud,1,m);    % u_basic with each eleement repeated r times
    uamp=abs(u_basic);
    phas=angle(u_basic);
    u=u_basic;
    if fcode==1
        ff=diag(f_basic);
        phas=2*pi*dt*cumsum(reshape(ao*ff,1,m))+phas;
        uexp=exp(j*phas);
        u=uamp.*uexp;
    end
end

tscale=[0:length(uamp)-1]/r;
tscale1=[0 0:length(uamp)-1 length(uamp)-1]/r;
dphas=[NaN diff(phas)]*r/2/pi;

figure(sigfig), clf, hold off % plot the signal parameters
subplot(3,1,1)
plot(tscale1,[0 abs(uamp) 0],'k','linewidth',1.5)
ylabel(' Amplitude ')
titlest=presetvalues(get(preset,'value')).Name;
title(titlest);
axis([-inf inf 0 1.2*max(abs(uamp))])

subplot(3,1,2)
plot(tscale, phas,'k','linewidth',1.5)
axis([-inf inf -inf inf])
ylabel(' phase [rad] ')

subplot(3,1,3)
plot(tscale,dphas*ceil(max(tscale)),'k','linewidth',1)
axis([-inf inf -inf inf])
xlabel(' \itt / t_b ')
ylabel(' \itf * Mt_b ')

% variables for ambigity calculations
t=tscale;
u=u;
