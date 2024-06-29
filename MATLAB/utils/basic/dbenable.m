function dbenable()
% enable all breakpoints

sDb = dbstatus();

for i = 1 : length(sDb)
    for j = 1 : length(sDb(i).expression)
        if strcmp(sDb(i).expression{j}, 'false')
            sDb(i).expression{j} = '';
        end
    end
end

end
