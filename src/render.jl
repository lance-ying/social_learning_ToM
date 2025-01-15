using PDDL
using PDDLViz, GLMakie

include("utils.jl")

# Define colors
vibrant = PDDLViz.colorschemes[:vibrant]
color_dict = Dict(
    :red => vibrant[1],
    :yellow => vibrant[2],
    :blue => colorant"#0072b2",
    :green => :springgreen,
    :none => :gray
)

# Define gem properties
gem_sides = [4, 4, 4, 4]
gem_angles = [pi, pi/4, 0.0, 0.0]
gem_shifts = [(0.0, 0.1), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
gem_sizes = [1.2, 1.2, 1.1, 1.0]
gem_colors = PDDLViz.colorschemes[:vibrantlight][[6,6, 6, 6]]

labels = Dict("1"=>"A", "2"=>"B", "3"=>"C", "4"=>"D")
# Construct gridworld renderer
RENDERER = PDDLViz.GridworldRenderer(
    resolution = (800, 800),
    has_agent = false,
    obj_renderers = Dict(
        :agent => (d, s, o) -> o.name ==:agent1 ? HumanGraphic(color = :red) : HumanGraphic(color = :blue),
        :wizard => (d, s, o) -> PDDLViz.WizGraphic(
            color=color_dict[get_obj_color(s, o).name]
        ),
        :door => (d, s, o) -> LockedDoorGraphic(
            # visible=s[Compound(:locked, [o])],
            color=color_dict[get_obj_color(s, o).name]
        ),
        :gem => (d, s, o) -> begin
            idx = parse(Int, string(o.name)[end])
            gem = MultiGraphic(GemGraphic(
                0.0, 0.0, gem_sizes[idx], gem_sides[idx], 1.0,
                visible=!(s[Compound(:offgrid, [o])] || s[Compound(:hidden, [o])]),
                color=gem_colors[idx]
            ),
            TextGraphic(
                labels[string(o.name)[end:end]], 0.0, -0.05, 0.4;
                color= :black,                        
                font=:bold
            ))
        end,
    ),
    obj_type_z_order = [:door, :wizard, :gem, :agent],
    show_inventory = false,
    # inventory_fns = [
    #     (d, s, o) -> s[Compound(:has, [Const(:human), o])]
    # ],
    inventory_types = [:item],
    inventory_labels = ["Inventory"],
    trajectory_options = Dict(
        :tracked_objects => [Const(:agent2)],
        :tracked_types => Const[],
        :object_colors => [:black]
    )
)
