function figs = create_figs(fig_kwargs, shape)
%create_figs creates figures with axes defined by <shape>.
%
% Arguments:
%   - shape - a repeating input of  2-element arrays.
%             <shape{i}>=[nrows, nclos] speciefies the subplots of the i-th
%             figure
% Returns:
%   - figs  - a struct of length(shape) with handles to the figure and all
%             its axes:
%             figs(i):
%               - fig
%               - axs(j)
%
% Example:
%   >> figs = create_figs([2,1], [3,1]);
%         figs(1)            figs(2)
%       |=========|        |=========|
%       | axs1(1) |        | axs2(1) |
%       |=========|        | axs2(2) |
%       | axs1(2) |        | axs2(3) |
%       |=========|        |=========|
%
arguments
    fig_kwargs = {}
end
arguments (Repeating)
    shape       % shape of subplots in the i-th figure
end

defaultFigureWindowStyle = get(0, "DefaultFigureWindowStyle");
set(0, "DefaultFigureWindowStyle", "docked")

if isempty(shape)
    shape = {[1, 1]};
end

nFigures = length(shape);

figs = struct();
for i = 1 : nFigures
    figs(i).fig = figure(fig_kwargs{:});

    axsShape = shape{i};
    h_tiled = tiledlayout(figs(i).fig, axsShape(1), axsShape(2), ...
        "TileSpacing", "compact", ...
        "Padding", "compact");
    for j = 1 : prod(axsShape)
        figs(i).axs(j) = nexttile(h_tiled, j);
        cla(figs(i).axs(j))
        hold(figs(i).axs(j), "on")
    end
end

set(0, "DefaultFigureWindowStyle", defaultFigureWindowStyle)

end
