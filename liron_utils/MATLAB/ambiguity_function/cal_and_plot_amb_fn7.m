% cal_and_plot_amb_fn7.m - Written by Eli Mozeson and Nadav Levanon
% calculates and plots the ambiguity function of a signal

% assumes that the work space includes a row vector u (signal complex envelope)
% and row vector t (time vector) where t is the equally spaced time points on
% which u is defined

% define the delay vector on which the ambiguity plot is calculated
dt=t(2)-t(1);    % dt is the sampling period of u(t)
m =length(t);    % total number of samples is u(t)
%T                 % normalized maximal delay (defined externally)
%N                    % number of grid points on each side of the delay axis (defined externally)

% calculate a delay vector with N+1 points that spans from zero delay to ceil(T*t(m))
% notice that the delay vector does not have to be equally spaced but must have all
% entries as integer multiples of dt

% two cases are possible
%         a) T*m>=N - the signal is oversampled relative to the delay axis definition
%         b) T*m<N  - the signal is undersampled (decrease N, increase T or increase r)
if T*m<N,
   msgbox(['N is too large, or r is too low for current definition of T.' ...
         'Using N=' sprintf('%d',ceil(T*m)) ' instead of N=' sprintf('%d',N)],'Warnning !!!');
   Nused=ceil(T*m);
else
   Nused=N;
end

dtau=ceil(T*m)*dt/Nused;
tau=round([0:1:Nused]*dtau/dt)*dt;

% df                 % spacing between adjacent grid points on the Doppler axis (defined externally)
% calculate K+1 equally spaced grid points of Doppler axis with df spacing
f=[0:1:K]*df;

% duplicate Doppler axis to show also negative dopplers (0 Doppler is calculated twice)
f=[-fliplr(f) f];

% calculate ambiguity function using sparse matrix manipulations (no loops)

% define a sparse matrix based on the signal samples u1 u2 u3 ... um
% with size m+ceil(T*m) by m (notice that u' is the conjugate transpoze of u)
% where the top part is diagonal (u*) on the diagonal and the bottom part is a zero matrix
%
%            [u1*  0   0  0 ...  0  ]
%            [ 0  u2*  0  0 ...  0  ]
%            [ 0   0  u3* 0 ...  0  ]    m rows
%            [ .                 .      .  ]
%            [ .                 .      .  ]
%            [ .   0   0     . ...  um*]
%            [ 0                      0  ]
%            [ .                      .  ]   Nused rows
%            [ 0   0   0  0 ...  0  ]
%
mat1=spdiags(u',0,m+ceil(T*m),m);

% define a convolution sparse matrix based on the signal samples u1 u2 u3 ... um
% where each row is a time(index) shifted versions of u.
% each row is shifted tau/dt places from the first row
% the minimal shift (first row) is zero
% the maximal shift (last row) is ceil(T*m) places
% the total number of rows is Nused+1
% number of colums is m+ceil(T*m)

% for example, when tau/dt=[0 2 3 5 6] and Nused=4
%
%            [u1 u2 u3 u4  ...               ... um  0  0  0  0  0  0]
%            [ 0  0 u1 u2 u3 u4  ...               ... um  0  0  0  0]
%            [ 0  0  0 u1 u2 u3 u4  ...               ... um  0  0  0]
%             [ 0  0  0  0  0 u1 u2 u3 u4  ...               ... um  0]
%            [ 0  0  0  0  0  0 u1 u2 u3 u4  ...               ... um]
%

% define a row vector with ceil(T*m)+m+ceil(T*m) places by padding u with zeros on both sides
u_padded=[zeros(1,ceil(T*m)),u,zeros(1,ceil(T*m))];

% define column indexing and row indexing vectors
cidx=[1:m+ceil(T*m)];
ridx=round(tau/dt)';

% define indexing matrix with Nused+1 rows and m+ceil(T*m) columns
% where each element is the index of the correct place in the padded version of u
index = cidx(ones(Nused+1,1),:) + ridx(:,ones(1,m+ceil(T*m)));

% calculate matrix
mat2 = sparse(u_padded(index));

% calculate the ambiguity matrix for positive delays given by
%
%    [u1 u2 u3 u4  ...               ... um  0  0  0  0  0  0] [u1*  0   0  0 ...  0  ]
%    [ 0  0 u1 u2 u3 u4  ...               ... um  0  0  0  0] [ 0  u2*  0  0 ...  0  ]
%    [ 0  0  0 u1 u2 u3 u4  ...               ... um  0  0  0]*[ 0   0  u3* 0 ...  0  ]
%     [ 0  0  0  0  0 u1 u2 u3 u4  ...               ... um  0] [ .                 .       .  ]
%    [ 0  0  0  0  0  0 u1 u2 u3 u4  ...               ... um] [ .                 .       .  ]=
%                                                            [ .   0   0  . ...  um*]
%                                                                              [ 0                     0  ]
%                                                                                 [ .                       .  ]
%                                                                [ 0   0   0  0 ...  0  ]
%
% where there are m columns and Nused+1 rows and each element gives an element
% of multiplication between u and a time shifted version of u*. each row gives
% a different time shift of u* and each column gives a different entry in u.
%
%
uu_pos=mat2*mat1;

clear mat2 mat1

% calcualte exponent matrix for full calculation of ambiguity function. the exponent
% matrix is 2*(K+1) rows by m columns where each row represents a possible Doppler and
% each column stands for a differnt place in u.
e=exp(-j*2*pi*f'*t);

% calcualte ambiguity function for positive delays by calcualting the integral for each
% possible delay and Doppler over all entries in u.
% a_pos has 2*(K+1) rows (Doppler) and Nused+1 columns (Delay)
a_pos=abs(e*uu_pos');

% normalize ambiguity function to have a maximal value of 1
a_pos=a_pos/max(max(a_pos));

% use the symetry properties of the ambiguity function to transform the negative Doppler
% positive delay part to negative delay, positive doppler
a=[flipud(conj(a_pos(1:K+1,:))) fliplr(a_pos(K+2:2*K+2,:))];

% define new delay and Doppler vectors
delay=[-fliplr(tau) tau];
freq=f(K+2:2*K+2)*ceil(max(t));

% exclude the zero Delay that was taken twice
delay=[delay(1:Nused) delay((Nused+2):2*(Nused+1))];
a=a(:,[1:Nused (Nused+2):2*(Nused+1)]);

% plot the ambiguity function and autocorelation cut
[amf amt]=size(a);

% create an all blue color map
cm=zeros(64,3);
% cm(:,3)=ones(64,1);

figure(ambfig), clf, hold off
mesh(delay, [0 freq], [zeros(1,amt);a])

hold on
surface(delay, [0 0], [zeros(1,amt);a(1,:)])

colormap(cm)
view(-40,50)
axis([-inf inf -inf inf 0 1])
xlabel(' {\it\tau}/{\itt_b}','Fontsize',12);
ylabel(' {\it\nu}*{\itMt_b}','Fontsize',12);
zlabel(' |{\it\chi}({\it\tau},{\it\nu})| ','Fontsize',12);
title(titlest);
hold off
