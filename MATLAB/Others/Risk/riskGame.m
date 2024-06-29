a=input('Number of attackers: ');

if isempty(a) 
    a=200;
end

b=input('Number of defenders: ');

if isempty(b) 
    b=1;
end

risk2(a,b);