function dbdisable()
% disable (but not remove) all (non-conditional) breakpoints

sDb = dbstatus();

for i = 1 : length(sDb)
    for j = 1 : length(sDb(i).expression)
        if isempty(sDb(i).expression{j})
            sDb(i).expression{j} = 'false';
        end
    end
end

end
