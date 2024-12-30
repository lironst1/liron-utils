function plot_ops(ax, options)
arguments
    ax
    options.title           string  = []
    options.xlabel          string  = []
    options.ylabel          string  = []
    options.zlabel          string  = []
    options.grid    (1,1)   string  = "on"
    options.axis            string  = []
    options.shading         string  = []
    options.xlim            double  = []
    options.ylim            double  = []
    options.zlim            double  = []
    options.legend          logical = false
    options.view            double  = []

    options.filename        string  = []
    options.format  (1,1)   string  = ".png"
    options.closefig        logical = false
    
    options.title_kwargs    cell    = {"FontSize", 18}
    options.label_kwargs    cell    = {"FontSize", 14}
end

if ax.Type == "axes"  % in case user wants to save figure, a figure handle can also be passed
    if ~isempty(options.title)
        title(ax, options.title, options.title_kwargs{:})
    end
    if ~isempty(options.xlabel)
        xlabel(ax, options.xlabel, options.label_kwargs{:})
    end
    if ~isempty(options.ylabel)
        ylabel(ax, options.ylabel, options.label_kwargs{:})
    end
    if ~isempty(options.zlabel)
        zlabel(ax, options.zlabel, options.label_kwargs{:})
    end
    if ~isempty(options.grid)
        grid(ax, options.grid)
    end
    if ~isempty(options.axis)
        axis(ax, options.axis)
    end
    if ~isempty(options.shading)
        shading(ax, options.shading)
    end
    if ~isempty(options.xlim)
        xlim(ax, options.xlim)
    end
    if ~isempty(options.ylim)
        ylim(ax, options.ylim)
    end
    if ~isempty(options.zlim)
        zlim(ax, options.zlim)
    end
    if options.legend
        legend(ax, options.label_kwargs{:})
    end
    if options.view
        view(ax, options.view)
    end
end


if ~isempty(options.filename)
    fig = ancestor(ax, "figure");

    [dirname, filename, format] = fileparts(options.filename);
    dirname = fullfile(dirname, "figs");
    if ~exist(dirname, "dir")
        mkdir(dirname)
    end
    if strlength(format) > 0
        options.format = format;
    end
    options.filename = fullfile(dirname, filename + options.format);

    exportgraphics(fig, options.filename)

    if options.closefig
        close(fig)
    end

end

end
