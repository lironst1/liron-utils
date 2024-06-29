function ll(dirname)
arguments
    dirname = PATHS().codeProjects
end

paths = string(properties(PATHS()));

tf = strcmpi(paths, dirname);
if any(tf)
    dirname = paths(tf);
    cd(PATHS().(dirname))
else
    cd(dirname)
end

end
