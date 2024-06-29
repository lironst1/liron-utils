classdef PATHS
    %PATHS

    properties (Constant, Hidden)

        pc = struct( ...
            "codeProjects", "C:\Users\liron\PyCharmProjects", ...
            "matlab", "C:\Users\liron\PyCharmProjects\Home\MATLAB", ...
            "matlabDrive", "C:\Users\liron\MATLAB Drive", ...
            "breakpoints", fullfile(prefdir, "breakpoints.mat") ...
            )
        mac = struct( ...
            "codeProjects", "/Users/lironst/Code Projects", ...
            "matlab", "/Users/lironst/Code Projects/Home/MATLAB", ...
            "matlabDrive", "", ...
            "breakpoints", fullfile(prefdir, "breakpoints.mat") ...
            )

    end

    properties

        codeProjects
        matlab
        matlabDrive
        breakpoints

    end

    methods

        function obj = PATHS()
            os = "mac";
            if ispc()
                os = "pc";
            end

            for field = string(fieldnames(obj)).'
                obj.(field) = obj.(os).(field);
            end
        end


    end

end
