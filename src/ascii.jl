# Functions for generating gridworld PDDL problems
using PDDL

"Converts ASCII gridworlds to PDDL problem."
function ascii_to_pddl(
    str::AbstractString,
    key_set::Union{AbstractString, Nothing} = nothing,
    name="doors-keys-gems-problem";
    key_dict = Dict(
        'r' => pddl"(red)",
        'b' => pddl"(blue)",
        'e' => pddl"(blue)"
    ),
    door_dict = Dict(
        'R' => pddl"(red)",
        'B' => pddl"(blue)",
        'Y' => pddl"(yellow)" ,
        'E' => pddl"(green)",
    )
)
    objects = Dict(
        :door => Const[], :key => Const[], :gem => Const[],
        :box => Const[], :color => Const[], :agent => Const[pddl"(agent1)", pddl"(agent2)"]
    )

    # Parse width and height of grid
    rows = split(str, "\n", keepempty=false)
    width, height = maximum(length.(strip.(rows))), length(rows)
    walls = parse_pddl("(= walls (new-bit-matrix false $height $width))")

    # Parse wall, item, and agent locations
    init = Term[walls]
    init_agent1 = Term[]
    init_agent2 = Term[]
    goal = pddl"(true)"
    for (y, row) in enumerate(rows)
        for (x, char) in enumerate(strip(row))
            if char == '.' # Unoccupied
                continue
            elseif char == 'W' # Wall
                wall = parse_pddl("(= walls (set-index walls true $y $x))")
                push!(init, wall)
            elseif char == 'C' # Box
                n = length(objects[:box]) + 1
                b = Const(Symbol("box$n"))
                push!(objects[:box], b)
                append!(init, parse_pddl("(= (xloc $b) $x)", "(= (yloc $b) $y)"))
                push!(init, parse_pddl("(closed $b)"))
                n = length(objects[:key]) + 1
                k = Const(Symbol("key$n"))
                # Check if key set is provided
                isnothing(key_set) && continue
                # Add key associated with box
                push!(objects[:key], k)
                n_boxes = length(objects[:box])
                if key_set[n_boxes] == 'C' # Box is empty
                    append!(init, parse_pddl("(= (xloc $k) -1)", "(= (yloc $k) -1)"))
                    push!(init, parse_pddl("(offgrid $k)"))
                else # Box has key
                    color = key_dict[key_set[n_boxes]]
                    push!(objects[:color], color)
                    append!(init, parse_pddl("(= (xloc $k) $x)", "(= (yloc $k) $y)"))
                    push!(init, parse_pddl("(iscolor $k $color)"))
                    push!(init, parse_pddl("(hidden $k)"))
                    push!(init, parse_pddl("(inside $k $b)"))
                end
            elseif haskey(door_dict, char) # Door
                n = length(objects[:door]) + 1
                d = Const(Symbol("door$n"))
                color = door_dict[char]
                push!(objects[:door], d)
                push!(objects[:color], color)
                append!(init, parse_pddl("(= (xloc $d) $x)", "(= (yloc $d) $y)"))
                push!(init, parse_pddl("(iscolor $d $color)"))
                push!(init, parse_pddl("(locked $d)"))
            elseif haskey(key_dict, char) # Key
                n = length(objects[:key]) + 1
                k = Const(Symbol("key$n"))
                color = key_dict[char]
                push!(objects[:key], k)
                push!(objects[:color], color)
                append!(init, parse_pddl("(= (xloc $k) $x)", "(= (yloc $k) $y)"))
                push!(init, parse_pddl("(iscolor $k $color)"))
            elseif char == 'g' || char == 'G' # Gem
                n = length(objects[:gem]) + 1
                g = Const(Symbol("gem$n"))
                push!(objects[:gem], g)
                append!(init, parse_pddl("(= (xloc $g) $x)", "(= (yloc $g) $y)"))
                if char == 'G'
                    goal = parse_pddl("(has agent1 $g)")
                end
            elseif char == 'M' # Agent
                append!(init_agent1, parse_pddl("(= (xloc agent1) $x)", "(= (yloc agent1) $y)"))

            elseif char == 'O' # Agent
                append!(init_agent2, parse_pddl("(= (xloc agent2) $x)", "(= (yloc agent2) $y)"))
            end
        end
    end
    append!(init, init_agent1)
    append!(init, init_agent2)

    # Create object list
    objlist = Const[]
    for objs in values(objects)
        sort!(unique!(objs), by=string)
        append!(objlist, objs)
    end
    # Create object type dictionary
    objtypes = Dict{Const, Symbol}()
    for (type, objs) in objects
        objs = unique(objs)
        for o in objs
            objtypes[o] = type
        end
    end

    problem = GenericProblem(Symbol(name), Symbol("doors-keys-gems"),
                             objlist, objtypes, init, goal,
                             nothing, nothing)
    return problem
end

function load_ascii_problem(path::AbstractString, keyset=nothing)
    str = open(f->read(f, String), path)
    return ascii_to_pddl(str, keyset)
end

function convert_ascii_problem(path::String)
    str = open(f->read(f, String), path)
    str = ascii_to_pddl(str)
    new_path = splitext(path)[1] * ".pddl"
    write(new_path, write_problem(str))
    return new_path
end

function get_filenames()
    path = "/Users/lance/Documents/GitHub/TomProjects.jl/watch_explore/dataset/problems/"
    filenames = readdir(path)
    return filenames
end

filenames = get_filenames()

for filename in filenames
    if endswith(filename, ".txt") and startswith(filename, "s")
        new_filename = filename[1:2]*filename[4]*filename[6:end]
        mv("/Users/lance/Documents/GitHub/TomProjects.jl/watch_explore/dataset/problems/" * filename,
           "/Users/lance/Documents/GitHub/TomProjects.jl/watch_explore/dataset/problems/" * new_filename)
    end
end

for filename in filenames
    if endswith(filename, ".txt")
        print(filename[1:end-4])
        print("\n")
        convert_ascii_problem("/Users/lance/Documents/GitHub/TomProjects.jl/watch_explore/dataset/problems/"*filename)
    end
end