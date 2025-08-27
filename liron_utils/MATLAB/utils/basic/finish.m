function finish()

sBreakpoints = dbstatus();

if ~isempty(sBreakpoints)
    save(PATHS().breakpoints, "sBreakpoints")
    fprintf("Breakpoints saved.\n")
else
    fprintf("No breakpoints to save.\n")
end

end
