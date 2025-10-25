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
        'E' => pddl"(green)",
    )
)
    objects = Dict(
        :door => Const[], :key => Const[], :gem => Const[],
        :wizard => Const[], :color => Const[], :agent => Const[]  # Start empty, add dynamically
    )

    # Parse width and height of grid
    # Stop at empty line (metadata starts after empty line)
    all_rows = split(str, "\n", keepempty=true)
    rows = []
    for row in all_rows
        if strip(row) == ""
            break  # Stop at empty line
        end
        push!(rows, row)
    end
    width, height = maximum(length.(strip.(rows))), length(rows)
    walls = parse_pddl("(= walls (new-bit-matrix false $height $width))")

    # Parse wall, item, and agent locations
    init = Term[walls]
    init_agent1 = Term[]
    init_agent2 = Term[]
    init_agent3 = Term[]
    goal = pddl"(true)"
    for (y, row) in enumerate(rows)
        for (x, char) in enumerate(strip(row))
            if char == '.' # Unoccupied
                continue
            elseif char == 'W' # Wall
                wall = parse_pddl("(= walls (set-index walls true $y $x))")
                push!(init, wall)
            elseif haskey(key_dict, char) # Box
                n = length(objects[:wizard]) + 1
                w = Const(Symbol("wizard$n"))
                push!(objects[:wizard], w)
                append!(init, parse_pddl("(= (xloc $w) $x)", "(= (yloc $w) $y)"))
                # push!(init, parse_pddl("(closed $b)"))
                n = length(objects[:key]) + 1
                k = Const(Symbol("key$n"))
                # Check if key set is provided
                # isnothing(key_set) && continue
                # Add key associated with box
                
                
                n_wizards = length(objects[:wizard])
            
                if char == 'e' # Box is empty

                    color = key_dict[char]
                    # append!(init, parse_pddl("(= (xloc $k) -1)", "(= (yloc $k) -1)"))
                    # push!(init, parse_pddl("(offgrid $k)"))
                    push!(init, parse_pddl("(iscolor $w $color)"))
                elseif char == 'b' # Box is occupied
                    push!(objects[:key], k)
                    color = key_dict[char]
                    push!(objects[:color], color)
                    append!(init, parse_pddl("(= (xloc $k) $x)", "(= (yloc $k) $y)"))
                    push!(init, parse_pddl("(iscolor $k $color)"))
                    push!(init, parse_pddl("(iscolor $w $color)"))
                    push!(init, parse_pddl("(hold $w $k)"))

                elseif char == 'r' # Box is occupied
                    push!(objects[:key], k)
                    color = key_dict[char]
                    push!(objects[:color], color)
                    append!(init, parse_pddl("(= (xloc $k) $x)", "(= (yloc $k) $y)"))
                    push!(init, parse_pddl("(iscolor $k $color)"))
                    push!(init, parse_pddl("(iscolor $w $color)"))
                    push!(init, parse_pddl("(hold $w $k)"))
                end
            elseif haskey(door_dict, char) # Door
                n = length(objects[:door]) + 1
                d = Const(Symbol("door$n"))
                color = door_dict[char]
                push!(objects[:door], d)
                push!(objects[:color], color)
                append!(init, parse_pddl("(= (xloc $d) $x)", "(= (yloc $d) $y)"))
                push!(init, parse_pddl("(iscolor $d $color)"))
                # push!(init, parse_pddl("(locked $d)"))
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
            elseif char == 'M' # Agent 1 (Observer/Player)
                if pddl"(agent1)" ∉ objects[:agent]
                    push!(objects[:agent], pddl"(agent1)")
                end
                append!(init_agent1, parse_pddl("(= (xloc agent1) $x)", "(= (yloc agent1) $y)"))

            elseif char == 'X' # Agent 2 (for inference)
                if pddl"(agent2)" ∉ objects[:agent]
                    push!(objects[:agent], pddl"(agent2)")
                end
                append!(init_agent2, parse_pddl("(= (xloc agent2) $x)", "(= (yloc agent2) $y)"))

            elseif char == 'Y' # Agent 3 (for inference)
                if pddl"(agent3)" ∉ objects[:agent]
                    push!(objects[:agent], pddl"(agent3)")
                end
                append!(init_agent3, parse_pddl("(= (xloc agent3) $x)", "(= (yloc agent3) $y)"))
            end
        end
    end
    append!(init, init_agent1)
    append!(init, init_agent2)
    append!(init, init_agent3)

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

# Commented out automatic execution - uncomment if needed for batch conversion
# function get_filenames()
#     path = "/Users/lance/Documents/GitHub/ObserveMove/dataset/problems_exp1/"
#     filenames = readdir(path)
#     return filenames
# end

# filenames = get_filenames()

# for filename in filenames
#     if endswith(filename, ".pddl") && startswith(filename, "s")
#         new_filename = filename[1:2]*filename[4]*filename[6:end]
#         mv("/Users/lance/Documents/GitHub/TomProjects.jl/watch_explore/dataset/problems/" * filename,
#            "/Users/lance/Documents/GitHub/TomProjects.jl/watch_explore/dataset/problems/" * new_filename)
#     end
# end

# for filename in filenames
#     if endswith(filename, ".txt")
#         print(filename[1:end-4])
#         print("\n")
#         convert_ascii_problem("/Users/lance/Documents/GitHub/ObserveMove/dataset/problems_exp1/"*filename)
#     end
# end