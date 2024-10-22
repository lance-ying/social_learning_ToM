using PDDL, SymbolicPlanners
using PDDLViz, GLMakie
using JSON3
using Random

PDDLViz.WizGraphic


include("src/render.jl")
include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("paths_new.jl")

PDDL.Arrays.register!()

"Generates stimulus animation from a initial state and plan."
function generate_stim_anim(
    path::Union{AbstractString, Nothing},
    domain::Domain,
    state::State,
    plan::AbstractVector{<:Term};
    renderer = RENDERER,
    trail_length = 5,
    show_inventory = true,
    inventory_titlesize = 28,
    framerate = 3,
    format = "gif",
    loop = -1,
    kwargs...
)
    # Render initial state
    canvas = PDDLViz.new_canvas(renderer)
    anim_initialize!(canvas, renderer, domain, state;
                     show_inventory, trail_length, kwargs...)
    # Set inventory title font size 
    if show_inventory
        canvas.blocks[2].titlesize = inventory_titlesize
    end
    # Animate plan
    anim = anim_plan!(canvas, renderer, domain, state, plan;
                      trail_length, show_inventory,
                      framerate, format, loop, kwargs...)
    # Save animation
    if !isnothing(path)
        save(path, anim)
    end
    return anim
end

function generate_stim_anim(
    path::Union{AbstractString, Nothing},
    domain::Domain,
    problem::Problem,
    plan::AbstractVector{<:Term};
    kwargs...
)
    state = initstate(domain, problem)
    anim = generate_stim_anim(path, domain, state, plan; kwargs...)
    return anim
end

"Create storyboard plot from a stimulus."
function generate_stim_storyboard(
    domain::Domain,
    problem::Problem,
    plan::AbstractVector{<:Term},
    timesteps::AbstractVector{Int};
    subtitles = fill("", length(timesteps)),
    xlabels = fill("", length(timesteps)),
    xlabelsize = 20, subtitlesize = 24, n_rows = 1,
    kwargs...
)
    # Generate animation without smooth transitions
    anim = generate_stim_anim(nothing, domain, problem, plan; kwargs...)
    # Create goal inference storyboard
    timesteps = timesteps .+ 1
    storyboard = render_storyboard(
        anim, timesteps;
        subtitles, xlabels, xlabelsize, subtitlesize, n_rows
    )
    return storyboard
end

"Generates animation segments from an initial state, plan, and splitpoints."
function generate_stim_anim_segments(
    basepath::Union{AbstractString, Nothing},
    domain::Domain,
    state::State,
    plan::AbstractVector{<:Term},
    splitpoints::AbstractVector{Int};
    renderer = RENDERER,
    trail_length = 5,
    show_inventory = true,
    inventory_titlesize = 28,
    framerate = 3,
    format = "gif",
    loop = -1,
    track_stopmarker = ' ',
    kwargs...
)
    trajectory = PDDL.simulate(domain, state, plan)
    # Adjust splitpoints to include the start and end of the trajectory
    splitpoints = copy(splitpoints) .+ 1
    if isempty(splitpoints) || last(splitpoints) != length(trajectory)
        push!(splitpoints, length(trajectory))
    end
    pushfirst!(splitpoints, 1)
    # Render initial state
    canvas = PDDLViz.new_canvas(renderer)
    anim_initialize!(canvas, renderer, domain, state;
                     show_inventory, trail_length,
                     track_stopmarker, kwargs...)
    # Set inventory title font size 
    if show_inventory
        canvas.blocks[2].titlesize = inventory_titlesize
    end
    # Animate trajectory segments between each splitpoint
    anims = PDDLViz.Animation[]
    for i in 1:length(splitpoints)-1
        a, b = splitpoints[i:i+1]
        anim = anim_trajectory!(canvas, renderer, domain, trajectory[a:b];
                                framerate, format, loop, show=false,
                                trail_length, track_stopmarker, kwargs...)
        if !isnothing(basepath)
            save("$(basepath)_$i.$format", anim)
        end
        push!(anims, anim)
    end
    return anims
end

function generate_stim_anim_segments(
    basepath::Union{AbstractString, Nothing},
    domain::Domain,
    problem::Problem,
    plan::AbstractVector{<:Term},
    splitpoints::AbstractVector{Int};
    kwargs...
)
    state = initstate(domain, problem)
    anims = generate_stim_anim_segments(
        basepath, domain, state, plan, splitpoints; kwargs...
    )
    return anims
end

"Generate stimuli JSON metadata."
function generate_stim_json(
    name::String,
    problem::Problem,
    splitpoints::AbstractVector{Int},
    statements::AbstractVector{<:AbstractString},
    key_colors::AbstractVector{<:Term} = Const[],
    plan_boxes::AbstractVector{<:Term} = Const[]
)
    times = splitpoints .+ 1
    n_images = length(splitpoints) + 1
    images = ["stimuli/segments/p$(name)_$i.gif" for i in 1:n_images]
    goal = parse(Int, string(problem.goal.args[2])[end])
    key_colors = string.(key_colors)
    box_ids = [parse(Int, string(box)[end]) for box in plan_boxes]
    json = (
        name = name,
        goal = goal,
        images = images,
        times = times,
        statements = statements,
        relevant_colors = key_colors,
        relevant_boxes = box_ids,
        length = length(splitpoints),
    )
    return json
end

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans")
STATEMENT_DIR = joinpath(@__DIR__, "dataset", "statements")
STIMULI_DIR = joinpath(@__DIR__, "dataset", "stimuli")

# Load domain
domain = load_domain(joinpath(@__DIR__, "dataset", "domain_render.pddl"))

## Generate animations for single plan / stimulus

# Load problem
problem_path = joinpath(PROBLEM_DIR, "s341.pddl")
plan = paths["s221_blue_exp"]
problem = load_problem(problem_path)
state = initstate(domain, problem)
# Load plan
plan_path = joinpath(PLAN_DIR, "demo.pddl")

plan = @pddl("(down agent2)","(down agent2)","(down agent2)","(down agent2)","(down agent2)")
plan, _, splitpoints = load_plan(plan_path)
p_id = splitext(basename(plan_path))[1]

canvas = PDDLViz.new_canvas(RENDERER)
anim_initialize!(canvas, RENDERER, domain, state)
anim = anim_plan!(canvas, RENDERER, domain, state, plan[1:25]; trail_length = 15, show_inventory=true)
save("/Users/lance/Documents/GitHub/ObserveMove/dataset/stimuli/full/demo.gif", anim)


for k in collect(keys(paths))[4:end]


    if endswith(k,"other_naive")
        continue
    end

    # if !startswith(k,"s53")
    #     continue
    # end

    name = k
    pid = name[1:4]
    println("Generating stimuli for $(name) ...")

    problem_path = joinpath(PROBLEM_DIR, "$(pid).pddl")
    problem = load_problem(problem_path)
    state = initstate(domain, problem)

    canvas = PDDLViz.new_canvas(RENDERER)

    plan = paths[name]

    canvas = PDDLViz.new_canvas(RENDERER)
    anim_initialize!(canvas, RENDERER, domain, state)
    print(plan)
    anim = anim_plan!(canvas, RENDERER, domain, state, plan; trail_length = 15, framerate = 3, format = "gif",loop = -1)
    save("/Users/lance/Documents/GitHub/ObserveMove/dataset/stimuli/$(name).gif", anim)

end
