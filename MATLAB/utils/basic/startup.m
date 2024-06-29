function startup()

ll

% format
format compact

% restore breakpoints
filename = PATHS().breakpoints;
if exist(filename, "file")
    load(filename, "sBreakpoints")
    delete(filename)

    dbstop(sBreakpoints)
    
    fprintf("Breakpoints restored.\n")
    dbstatus()
else
    fprintf("Breakpoints not restored.\n")
end

end
