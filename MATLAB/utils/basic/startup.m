function startup()

% warnings
% warning("on", "verbose")

% format
format compact

boot()

% restore breakpoints
filename = PATHS().breakpoints;
if exist(filename, "file")
    load(filename, "sBreakpoints")
    delete(filename)

    dbstop(sBreakpoints)
    
    dbstatus()
else
    fprintf("Breakpoints not restored.\n")
end

end
