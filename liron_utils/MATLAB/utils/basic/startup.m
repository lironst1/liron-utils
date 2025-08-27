function startup()

set_defaults()

% Restore breakpoints
filename = PATHS().breakpoints;
if exist(filename, "file")
    load(filename, "sBreakpoints")
    delete(filename)

    dbstop(sBreakpoints)

    dbstatus()
else
    fprintf("Breakpoints not restored.\n")
end

boot()

end
