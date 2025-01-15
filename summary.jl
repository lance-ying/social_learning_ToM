model = "mentalize"
PLAN_DIR = joinpath(@__DIR__, "results", "plans", model)
num_steps = []
plan_cost = []

# num_steps = 0
action_cost = Dict(:move => 2, :interact => 5, :observe => 1.0)
# exp_pids = readdir(PLAN_DIR)

for p_id in exp_pids
    
        if occursin("naive",p_id)
            continue
        end
    
        plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, p_id*".pddl"))

        # num_steps += length(plan)

        push!(num_steps, length(plan))

        push!(plan_cost, calculate_plan_cost(plan, action_cost))

end

# println(num_steps/54)

using Statistics


println("Mean number of steps: ", mean(num_steps))
println("Standard deviation of steps: ", std(num_steps)/sqrt(length(num_steps)))

println("Mean plan cost: ", mean(plan_cost))
println("Standard deviation of plan cost: ", std(plan_cost)/sqrt(length(plan_cost)))

println(num_steps/54)
