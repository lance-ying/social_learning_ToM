using GenGPT3

EXAMPLE_TRANSLATIONS = """
Input: The player believes that there is a blue key in box 1.
Output: (believes player (exists (?k - key) (and (iscolor ?k blue) (inside ?k box1) ) ) )
Input: The player believes that there is no key in box 1, box 2, and box 3.
Output: (believes player (not (exists (?k - key) (or (inside ?k box1) (inside ?k box2) (inside ?k box3) ) ) ) )
Input: The player believes that there is a red key in either box 1 or box 2.
Output: (believes player (exists (?k - key) (and (iscolor ?k red) (or (inside ?k box1) (inside ?k box2) ) ) )
Input: The player believes that there is no red key in box 3.
Output: (believes player (not (exists (?k - key) (and (iscolor ?k red) (inside ?k box3) ) ) ) )
"""

"Translate belief statement from English to PDDL."
function translate_statement(
    statement::AbstractString,
    llm = GPT3GF(model="gpt-3.5-turbo-instruct", stop="\n",
                 max_tokens=512, temperature=0.0);
    examples::AbstractString = EXAMPLE_TRANSLATIONS,
    verbose::Bool = false
)
    verbose && println(statement)
    prompt = examples * "Input: " * statement * "\n" * "Output: "
    output = strip(llm(prompt))
    verbose && println(output)
    return parse_pddl(output)
end

"Translate belief statement dataset from English to PDDL."
function translate_statement_dataset(
    PLAN_IDS, STATEMENTS::Dict;
    llm = GPT3GF(model="gpt-3.5-turbo-instruct", stop="\n",
                 max_tokens=512, temperature=0.0),
    examples = EXAMPLE_TRANSLATIONS,
    verbose::Bool = true
)
    pddl_dataset = Dict{String, Vector{Term}}()
    for plan_id in PLAN_IDS
        verbose && println(plan_id)
        statements = STATEMENTS[plan_id]
        pddl_statements = map(statements) do stmt
            translate_statement(stmt, llm; examples, verbose)
        end
        pddl_dataset[plan_id] = pddl_statements
    end
    return pddl_dataset
end
