clear, clc
close all

global x t sineSum han

%% DEFINE PARAMETERS
% Mode-related parameters
numModes = 20;                              % Number of summed sines
modeIndex = 1:numModes;
% Location-related parameters
L = 1;                                      % Rope's length [m]
x = linspace(0, L, 5001);                   % x-axis location [m]
k = (pi/L) * modeIndex;                     % Wavenumber [rad/m]
% Time-related parameters
v = 1;                                      % Velocity (depending on the rope's elastic constant) [m/sec]
freq = (0.5*v/L) * modeIndex;               % Frequency [Hz]
omega = k * v;                              % Angular velocity [rad/sec]
T = 2*pi ./ omega;                          % Period [sec]   ***PERIOD SEEMS TO BE 12 TIMES LARGER...***
t = linspace(0, T(1), 1001);                % Time [sec]
% W = repmat(omega, length(x), 1);


%% DEFINE INITIAL CONDITIONS
% Here the user may choose the initial conditions of the system, including
% the rope's shape, [the initial velocity of each point in space and the
% rope's elastic constant].   *Note: Parameters in brackets are yet to be
% defined in the script.

% Triangular shape
for i = 1 : length(x)
    if x(i) <= mean(x)
        y(1,i) = 2*L * x(i);
    else
        y(1,i) = 2*L * (1-x(i));
    end
end

% Rectangular shape
% y(1,1:length(x)) = 1;

% Sine shape
% y = sin(k(1) * x);

rope_GUI;
han = findall(0,'Tag','axes1');

plot(han,x,y,'.'),   xlabel(han,'Location on Rope [m]'), ylabel(han,'Amplitude [m]'), ylim(han,[-1,1]), grid (han,'on')
hold (han,'on')   % ASK IN GUI 'Leave trace?'


%% Modes
M = sin(x' * k);                              % A 2-D matrix of amp in every point x and every mode (t=0)
M = M / norm(M);
x_mode = permute(M, [1,3,2]);

N = cos(t' * omega);                          % A 2-D matrix of amp in every time t and every mode (x=L/2)
t_mode = permute(N, [3,1,2]);

x_t_mode = x_mode .* t_mode;                  % A 3-D matrix of location x, time t and mode
A = y * M;
A = permute(A, [1,3,2]);
tmp = x_t_mode .* A;
sineSum = sum(tmp, 3);







% getValue('slider1') 
% 
% sliderHandle=findall(0,'Tag','slider1');
% if ~empty(sliderHandle) 
%     while 0 
%     
%     get(sliderHandle,'Value') 
%     
%     drawnow, pause(0.4) 
%     
%     end 
% end
    


 
% % Find amplitude of m-th mode
% 
% % Represent the shape as a sum of sines (curve-fitting)
% sineSum_example = M * A';
% 
% plot (x, sineSum_example, '--')
% title('Sum of sines (t=0)')
% legend('y - Original Shape', '\psi - Sum of sines', 'location', 'northeast')


