using PDDL, Printf
using SymbolicPlanners, InversePlanning
using Gen, GenParticleFilters
using PDDLViz, GLMakie, CairoMakie
using PDDLViz.Makie.FileIO
using CSV, DataFrames

include("src/utils.jl")
include("src/render.jl")
include("src/plan_io.jl")

"Adds a subplot to a storyboard with a line plot of probabilities."
function storyboard_prob_lines!(
    fig_or_pos::Union{Figure, GridPosition, GridSubposition}, probs, ts=Int[];
    names = ["Series $i" for i in size(probs)[1]],
    colors = PDDLViz.colorschemes[:vibrant][1:length(names)],
    show_legend = false, ts_linewidth = 1, ts_fontsize = 24,
    legend_title = "Legend Title",
    xlabel = "Time", ylabel = "Probability", ylimits = (0, 1),
    upper = nothing, lower = nothing,
    ax_args = (), kwargs...
)
    if fig_or_pos isa Figure
        n_rows, n_cols = size(fig_or_pos.layout)
        width, height = size(fig_or_pos.scene)
        grid_pos = fig_or_pos[n_rows+1, 1:n_cols]
    else
        grid_pos = fig_or_pos
    end
    # Add probability subplot
    if length(ts) == size(probs)[2]
        curves = [[Makie.Point2f(t, p) for (t, p) in zip(ts, probs[i, :])]
                  for i in 1:size(probs)[1]]
        ax, _ = series(
            grid_pos, curves;
            color = colors, labels = names,
            axis = (xlabel = xlabel, ylabel = ylabel,
                    limits=((first(ts)-0.5, last(ts)+0.5), ylimits), ax_args...),
            kwargs...
        )
    else
        ax, _ = series(
            grid_pos, probs,
            color = colors, labels = names,
            axis = (xlabel = xlabel, ylabel = ylabel,
                    limits=((1, size(probs, 2)), ylimits), ax_args...),
            kwargs...
        )
    end
    # Add legend to subplot
    if show_legend
        axislegend(legend_title, framevisible=false)
    end
    # Add upper and lower bounds
    if !isnothing(upper) && !isnothing(lower)
        ts = ts == Int[] ? collect(1:size(probs, 2)) : ts
        for (j, (l, u)) in enumerate(zip(eachrow(lower), eachrow(upper)))
            bplt = band!(ax, ts, l, u, color=(colors[j], 0.2))
            translate!(bplt, 0, 0, -1)
        end
    end
    # Add vertical lines at timesteps
    if !isempty(ts)
        vlines!(ax, ts, color=:black, linestyle=:dash,
                linewidth=ts_linewidth)
    end
    # Resize figure to fit new plot
    if fig_or_pos isa Figure
        rowsize!(fig_or_pos.layout, 1, Auto(1.0))
        rowsize!(fig_or_pos.layout, n_rows+1, Auto(0.3))
        resize!(fig_or_pos, width, trunc(Int, height * 1.35))
        return fig_or_pos
    else
        return ax
    end
end

"Adds a subplot to a storyboard with a stacked bar plot of probabilities."
function storyboard_prob_bars!(
    fig_or_pos::Union{Figure, GridPosition, GridSubposition}, probs, ts=Int[];
    names = ["Series $i" for i in size(probs)[1]],
    colors = PDDLViz.colorschemes[:vibrant][1:length(names)],
    show_legend = false,
    legend_title = "Legend Title",
    xlabel = "Time", ylabel = "Probability", ylimits = (0, 1),
    upper = nothing, lower = nothing,
    ax_args = (), kwargs...
)
    if fig_or_pos isa Figure
        n_rows, n_cols = size(fig_or_pos.layout)
        width, height = size(fig_or_pos.scene)
        grid_pos = fig_or_pos[n_rows+1, 1:n_cols]
    else
        grid_pos = fig_or_pos
    end
    # Add probability subplot
    ts = ts == Int[] ? collect(1:size(probs, 2)) : ts
    colors = repeat(colors, length(ts))
    group = repeat(1:size(probs)[1], length(ts))
    starts = repeat(ts, inner=size(probs, 1)) .- 0.5
    stops = vec(probs) .+ starts
    ax, _ = barplot(
        grid_pos, group, stops, fillto = starts, direction = :x,
        color = colors, labels = names, 
        axis = (xlabel = xlabel, ylabel = ylabel, yreversed = true,
                limits=((first(ts)-0.5, last(ts)+0.5), (nothing, nothing)), ax_args...),
    )
    # Add upper and lower bounds
    if !isnothing(upper) && !isnothing(lower)
        new_lower = vec(lower) .+ starts
        new_upper = vec(upper) .+ starts
        rangebars!(ax, group, new_lower, new_upper,
                   color=:black, whiskerwidth=10, direction=:x)
    end
    # Add bar labels 
    bar_labels = [@sprintf("%.2f", p) for p in vec(probs)]
    bar_offset = 0.05    
    for (i, label) in enumerate(bar_labels)
        if probs[i] <= 0.6
            x = isnothing(upper) || isnothing(lower) ?
                    stops[i] + bar_offset : new_upper[i] + bar_offset
            text!(ax, x, group[i], text=label,
                  fontsize=28, align=(:left, :center), color=:black)
        else
            x = isnothing(upper) || isnothing(lower) ?
                    stops[i] - bar_offset : new_lower[i] - bar_offset
            text!(ax, x, group[i], text=label,
                  fontsize=28, align=(:right, :center), color=:white)
        end
    end
    # Add legend to subplot
    if show_legend
        axislegend(legend_title, framevisible=false)
    end
    # Add vertical lines at starts of barplots
    vlines!(ax, starts, color=:black, linestyle=:solid)
    # Resize figure to fit new plot
    if fig_or_pos isa Figure
        rowsize!(fig_or_pos.layout, 1, Auto(1.0))
        rowsize!(fig_or_pos.layout, n_rows+1, Auto(0.3))
        resize!(fig_or_pos, width, trunc(Int, height * 1.35))
        return fig_or_pos
    else
        return ax
    end
end

# Register PDDL array theory
PDDL.Arrays.register!()

# Load domain and problems
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans")
STATEMENT_DIR = joinpath(@__DIR__, "dataset", "statements")
RESULTS_DIR = joinpath(@__DIR__, "results")
DOMAIN = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

# Load plans
PLAN_IDS, PLANS, _, SPLITPOINTS = load_plan_dataset(PLAN_DIR)
_, STATEMENTS = load_statement_dataset(STATEMENT_DIR)

# Load human and model
model_path = joinpath(RESULTS_DIR, "models", "results_joint_uniform.csv")
human_path = joinpath(RESULTS_DIR, "humans", "stimuli_data.csv")
model_df = CSV.read(model_path, DataFrame)
human_df = CSV.read(human_path, DataFrame)

# Load problems
PROBLEMS = Dict{String, Problem}()
for path in readdir(PROBLEM_DIR)
    name, ext = splitext(path)
    ext == ".pddl" || continue
    PROBLEMS[name] = load_problem(joinpath(PROBLEM_DIR, path))
end
DOMAIN = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

# Set whetheter to plot bars or lines for goal probabilities
GOAL_PLOT_TYPE = :bars

# plan_index = "1_1"
for plan_index in PLAN_IDS
    # Load plan and judgment points
    plan = PLANS[plan_index]
    times = SPLITPOINTS[plan_index].+1

    # Filter out results for this plan stimuli
    filter_fn = r -> r.plan_id == plan_index && (r.is_judgment || r.timestep == 0)
    model_sub_df = filter(filter_fn, model_df)

    filter_fn = r -> r.plan_id == plan_index
    human_sub_df = filter(filter_fn, human_df)

    # Switch to CairoMakie for plotting
    CairoMakie.activate!()

    # Initialize state, and set renderer resolution to fit state grid
    state = initstate(DOMAIN, PROBLEMS[plan_index])
    grid = state[pddl"(walls)"]
    height, width = size(grid)
    RENDERER.resolution = (width * 100, (height + 1) * 100 + 50)
    RENDERER.inventory_labelsize = 48

    # Simulate final state
    trajectory = PDDL.simulate(DOMAIN, state, plan)
    # Render final state 
    canvas = render_state(RENDERER, DOMAIN, trajectory[end])
    ax = canvas.blocks[1]

    # Add key outline for any state where key is picked up
    for t in eachindex(plan)
        plan[t].name != :pickup && continue
        x, y = get_obj_loc(trajectory[t], pddl"(human)")
        key_graphic = PDDLViz.KeyGraphic(x, height-y+1, color=:grey90)
        PDDLViz.graphicplot!(ax, key_graphic)
    end

    # Render initial state marker
    canvas = render_trajectory!(
        canvas, RENDERER, DOMAIN, trajectory[1:1],
        track_stopmarker='⦿', track_markersize=0.6, object_colors = [(:blue, 0.75)]
    )
    # Render rest of trajectory
    canvas = render_trajectory!(
        canvas, RENDERER, DOMAIN, trajectory[2:end],
        track_stopmarker=' ', object_colors = [(:red, 0.75)],
        object_start_colors=[(:blue, 0.75)],
    )

    # Add tooltips at judgment points
    ax = canvas.blocks[1]
    tooltip_locs = []
    for (i, t) in enumerate(times[2:end])
        x, y = get_obj_loc(trajectory[t], pddl"(human)")
        y = height - y + 1
        act = t == times[end] ? plan[t-1] : plan[t]
        if act.name in (:up, :down)
            placement = (x == width || (x, y) in tooltip_locs) ? :left : :right
            tooltip!(ax, x, y, string(i), font = :bold, fontsize=40,
                placement=:right, textpadding = (12, 12, 4, 4))
        elseif (x, y) in tooltip_locs
            tooltip!(ax, x, y+0.1, string(i), font = :bold, fontsize=40,
                placement=:above, textpadding = (12, 12, 4, 4))
        elseif y == 1
            tooltip!(ax, x, y, string(i), font = :bold, fontsize=40,
                placement=:right, textpadding = (12, 12, 4, 4))
        else
            tooltip!(ax, x, y-0.1, string(i), font = :bold, fontsize=40,
                placement=:below, textpadding = (12, 12, 4, 4))
        end
        push!(tooltip_locs, (x, y))
    end
    canvas

    # Save stimulus image to temporary file
    img_path = joinpath(@__DIR__, "figures", "trajectory_p$(plan_index).png")
    save(img_path, canvas)

    # Create new figure with image as top plot
    figure = Figure(resolution = (1200, 2400))
    ax = Axis(figure[1, 1], aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax) 
    image!(ax, rotr90(load(img_path)))
    rowsize!(figure.layout, 1, Auto(1.0))

    # Decide whether to plot bars or lines
    goal_plot_f! = GOAL_PLOT_TYPE == :bars ? 
        storyboard_prob_bars! : storyboard_prob_lines!

    # Plot human goal inferences
    goal_probs = Matrix(human_sub_df[:, r"goal_probs_\d"])
    goal_colors = PDDLViz.colorschemes[:vibrant][[4, 5, 6, 8]]
    goal_linestyles = [:dash, :solid, :dashdot, :dashdotdot]

    goal_probs_sem = Matrix(human_sub_df[:, r"goal_probs_sem_\d+"])
    goal_probs_ci = 1.96 * goal_probs_sem
    upper = min.(goal_probs .+ goal_probs_ci, 1.0)
    lower = max.(goal_probs .- goal_probs_ci, 0.0)

    ax = goal_plot_f!(
        figure[2, 1][1, 1], goal_probs', collect(1:length(times[2:end])),
        upper = upper', lower = lower',
        names = ["Triangle", "Square", "Hexagon", "Circle"],
        colors = goal_colors,
        ts_linewidth = 4, ts_fontsize = 48,
        marker = :circle, markersize = 24, strokewidth = 1.0,
        linewidth = 10, linestyle = goal_linestyles,
        xlabel = "", ylabel = "Goal Ratings",
        ax_args = (
            xticks = collect(1:length(times[2:end])), 
            xticklabelsize = 48,
            ylabelsize = 48, yticklabelsize = 40,
            title="Humans",
            titlesize=52, titlealign=:left, titlefont=:regular
        )
    )
    if GOAL_PLOT_TYPE == :bars
        ax.yticks = ([1, 2, 3, 4], ["T", "S", "H", "C"])
        ax.ylabelpadding = 30
    end

    rowsize!(figure.layout, 2, Auto(0.20))
    figure

    # # Add error ribbons
    # if GOAL_PLOT_TYPE == :lines
    #     goal_probs_sem = Matrix(human_sub_df[:, r"goal_probs_sem_\d+"])
    #     goal_probs_ci = 1.96 * goal_probs_sem
    #     upper = min.(goal_probs .+ goal_probs_ci, 1.0)
    #     lower = max.(goal_probs .- goal_probs_ci, 0.0)
    #     for (j, (l, u)) in enumerate(zip(eachcol(lower), eachcol(upper)))
    #         bplt = band!(ax, collect(1:length(times[2:end])), l, u, color=(goal_colors[j], 0.2))
    #         translate!(bplt, 0, 0, -1)
    #     end
    #     figure
    # end

    # Plot model goal inferences
    goal_probs = Matrix(model_sub_df[:, r"goal_probs_\d"])
    goal_probs = goal_probs[2:end, :]
    ax = goal_plot_f!(
        figure[2, 1][1, 2], goal_probs', collect(1:length(times[2:end])),
        names = ["Triangle", "Square", "Hexagon", "Circle"],
        colors = goal_colors,
        ts_linewidth = 4, ts_fontsize = 48,
        marker = :circle, markersize = 24, strokewidth = 1.0,
        linewidth = 10, linestyle = goal_linestyles,
        xlabel = "", ylabel="P(Goal | Actions)",
        ax_args = (
            xticks = collect(1:length(times[2:end])), 
            xticklabelsize = 48, 
            ylabelsize = 48, yticklabelsize = 40,
            title="BToM Model",
            titlesize=52, titlealign=:left, titlefont=:regular
        )
    )
    if GOAL_PLOT_TYPE == :bars
        ax.yticks = ([1, 2, 3, 4], ["T", "S", "H", "C"])
        ax.ylabelpadding = 30
    end
    figure

    # Add legend for goal probabilities
    nrows, ncols = size(figure.layout)
    if GOAL_PLOT_TYPE == :bars
        elems = [PolyElement(;color) for color in goal_colors]
    else
        elems = [[LineElement(;color, linestyle, linewidth=10),
                  MarkerElement(;color, marker=:circle, markersize=36)]
                for (color, linestyle) in zip(goal_colors, goal_linestyles)]
    end
    Legend(figure[nrows + 1, :], elems,
        ["Triangle", "Square", "Hexagon", "Circle"],
        orientation=:horizontal, colgap=50,
        patchsize = (80, 30), labelsize = 36,
        framevisible=false, halign=:center)
    rowsize!(figure.layout, nrows + 1, Auto(0.1))
    figure

    # Plot human belief statement ratings
    statement_probs = Matrix(human_sub_df[:, r"statement_probs_\d"])
    statement_colors = Makie.colorschemes[:Egypt][[1, 2]]
    statement_linestyles = [:dash, :solid]

    statement_probs_sem = Matrix(human_sub_df[:, r"statement_probs_sem_\d+"])
    statement_probs_ci = 1.96 * statement_probs_sem
    upper = min.(statement_probs .+ statement_probs_ci, 1.0)
    lower = max.(statement_probs .- statement_probs_ci, 0.0)

    ax = storyboard_prob_lines!(
        figure[4, 1][1, 1], statement_probs', collect(1:length(times[2:end])),
        upper = upper', lower = lower',
        names = ["Statement 1", "Statement 2"],
        colors = statement_colors,
        ts_linewidth = 4, ts_fontsize = 48,
        marker = :circle, markersize = 24, strokewidth = 1.0,
        linewidth = 10, linestyle = statement_linestyles,
        xlabel = "", ylabel = "Belief Ratings",
        ax_args = (
            xticks = collect(1:length(times[2:end])), 
            xticklabelsize = 48,
            ylabelsize = 48, yticklabelsize = 40,
            title="Humans",
            titlesize=52, titlealign=:left, titlefont=:regular
        )
    )
    rowsize!(figure.layout, 4, Auto(0.20))
    figure

    # # Add error ribbons
    # statement_probs_sem = Matrix(human_sub_df[:, r"statement_probs_sem_\d+"])
    # statement_probs_ci = 1.96 * statement_probs_sem
    # upper = min.(statement_probs .+ statement_probs_ci, 1.0)
    # lower = max.(statement_probs .- statement_probs_ci, 0.0)
    # for (j, (l, u)) in enumerate(zip(eachcol(lower), eachcol(upper)))
    #     bplt = band!(ax, collect(1:length(times[2:end])), l, u, color=(statement_colors[j], 0.2))
    #     translate!(bplt, 0, 0, -1)
    # end
    # figure

    # Plot model belief statement ratings
    statement_probs = Matrix(model_sub_df[:, r"statement_probs_\d"])
    statement_probs = statement_probs[2:end, :]
    storyboard_prob_lines!(
        figure[4, 1][1, 2], statement_probs', collect(1:length(times[2:end])),
        names = ["Statement 1", "Statement 2"],
        colors = statement_colors,
        ts_linewidth = 4, ts_fontsize = 48,
        marker = :circle, markersize = 24, strokewidth = 1.0,
        linewidth = 10, linestyle = statement_linestyles,
        xlabel = "", ylabel="L(Bel | Actions)",
        ax_args = (
            xticks = collect(1:length(times[2:end])), 
            xticklabelsize = 48,
            ylabelsize = 48, yticklabelsize = 40,
            title="BToM Model",
            titlesize=52, titlealign=:left, titlefont=:regular
        )
    )
    figure

    # Add legend for statement probabilities
    nrows, ncols = size(figure.layout)
    elems = [LineElement(;color, linestyle, linewidth=10)
            for (color, linestyle) in zip(statement_colors, statement_linestyles)]
    labels = [s for (i, s) in enumerate(STATEMENTS[plan_index])]
    # Add line break in middle of each label
    labels = map(labels) do l
        words = split(l, " ")
        new_words = String[]
        count = 0
        for w in words
            if count <= 28
                count += length(w)
                if count > 28
                    push!(new_words, "\n")
                end
            end
            push!(new_words, w)
        end
        return join(new_words, " ")
    end
    Legend(figure[nrows + 1, :], elems, labels,
        patchsize = (30, 30), rowgap = 10, labelsize = 48,
        framevisible=false, halign=:left, tellwidth=false)
    rowsize!(figure.layout, nrows + 1, Auto(0.25))
    figure

    # Save figure
    fig_path = joinpath(@__DIR__, "figures", "storyboard_p$(plan_index)")
    save(fig_path * ".png", figure)
    save(fig_path * ".pdf", figure)
    display(figure)
end
