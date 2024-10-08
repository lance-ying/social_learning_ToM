# Grounding Language about Belief in a Bayesian Theory-of-Mind

A Bayesian inverse planning model of how people quantitatively attribute their confidence in statements about the beliefs of other agents.

## Setup

To set up the environment for this project, make sure the `belief_modeling` directory is set as the active environment. In VSCode with the Julia for VSCode extension, you can do this by right-clicking the directory in the file explorer, and selecting `Julia: Activate This Environment`. Alternatively, you can use the `activate` command in the Julia REPL's `Pkg` mode.

Then run the following commands in via `Pkg` mode in the Julia REPL:

```
add https://github.com/probcomp/InversePlanning.jl.git https://github.com/probcomp/GenGPT3.jl.git
instantiate
```

To use GenGPT3.jl, add your OpenAI API key as an environment variable named `OPENAI_API_KEY`. You can follow [this guide](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety), or set `ENV["OPENAI_API_KEY"]` to the value of your API key in the Julia REPL. To keep your API key secure, **do not** save its value within this repository.

## Project Structure

- The `dataset` directory contains all plans, problems, and stimuli
- The `src` directory contains non-top-level source files
- `run_experiment.jl` is the main experiment script
- `testbed.jl` is for experimenting with modeling and inference parameters
- `stimuli.jl` generates stimuli animations and metadata
