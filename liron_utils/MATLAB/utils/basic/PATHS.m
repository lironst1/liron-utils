classdef PATHS
    %PATHS
    % Define frequently used directories and move between them easily.
    % To define a new path:
    %   - Define a new property with a short name <prop>
    %   - Add its appropriate path in each OS
    %
    % Examples:
    %   1. Use the 'cdl' function:
    %       >> cdl downloads
    %
    %   2. Get path as string:
    %       >> path = PATHS().downloads;
    %       >> cd(path)

    properties (Constant, Hidden)

        pc = struct( ...
            "downloads",    "C:\Users\liron\Downloads", ...
            "code",         "C:\Users\liron\PyCharmProjects", ...
            "matlab",       "C:\Users\liron\PyCharmProjects\Home\liron-utils\liron_utils\MATLAB", ...
            "matlabDrive",  "C:\Users\liron\MATLAB Drive", ...
            "breakpoints",  fullfile(prefdir, "breakpoints.mat") ...
            )

        mac = struct( ...
            "downloads",    "/Users/lironst/Downloads", ...
            "code",         "/Users/lironst/Code Projects", ...
            "matlab",       "/Users/lironst/Code Projects/Home/liron-utils/liron_utils/MATLAB", ...
            "matlabDrive",  "", ...  % TODO
            "breakpoints",  fullfile(prefdir, "breakpoints.mat") ...
            )

    end

    properties

        downloads
        code
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

    methods (Static)

        function cdl(dirname)
            arguments
                dirname = PATHS().code
            end

            if isempty(dirname)
                dirname = PATHS().code;
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


    end

end
