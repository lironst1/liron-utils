function BP

% tasks
% generate list legal numbers
% draw number
%
% select next number to ask
%
% answer bull pgia per asked number
%
% based on an answer, update legal list
%

[n,d]=generateList;


%%  draw secret number

 h = waitbar(0,'Please wait...');

 Ngames=20;

for k = 1: Ngames
    waitbar(k/Ngames,h)

    Nsecret=n(randi(numel(n)));

    iter(k)= findNsecret (n,d, Nsecret);

end

delete(h)

figure (12)
hist(iter)
title('histogram of number of questions')

mean(iter)
median(iter)

end


function iter=findNsecret(n,d,Nsecret)


iter=1;
while true

    Nask=n(randi(numel(n)));

    [b,p]=bullPgia(Nask,Nsecret);


    if b==4
        % success
        if 0
        fprintf(1,'\n After %d questions, The secret number is %d \n\n',iter, Nask) ;
        end
        break
    end

    ok=[];
    for k=1:numel(n)
        [b1 p1]=bullPgia (n(k), Nask);
        ok(k)=all([b p] == [b1 p1]);
    end
    n=n(boolean(ok));

    if 0
    fprintf(1,'\n iter: %d Nask %d\n Nsecret %d\n b: %d   p: %d\n N: %d\n\n',iter, Nask,  Nsecret,  b , p , numel(n));
    end

    iter=iter+1;

end   % while true

end   % function findNsecret(n,d,Nsecret)

function [b,p]=bullPgia(Nask,Nsecret)

a=num2dig(Nask);
s=num2dig(Nsecret);

b=sum(a==s);

p=numel(intersect(a,s))  - b;

end %function [b,p]=bullPgia(Nask,Nsecret)


function [n,d]=generateList

n=1234:9999;

d=[];
ok=[];

for k=1:numel(n)
    d(k,:)=num2dig(n(k));
    ok(k) = numel(unique(d(k,:))) == 4;        % sum(ok)
end
ok=boolean(ok);

n=n(ok);
d=d(ok,:);

end

function d=num2dig(m)
d(1)=floor(m/1000);
m=m-d(1)*1000;
d(2)=floor(m/100);
m=m-d(2)*100;
d(3)=floor(m/10);
m=m-d(3)*10;
d(4)=m;
end
