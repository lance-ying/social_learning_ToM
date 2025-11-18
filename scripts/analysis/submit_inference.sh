#!/bin/bash
#SBATCH --job-name=inference_exp3
#SBATCH --output=inference_exp3_%j.out
#SBATCH --error=inference_exp3_%j.err
#SBATCH --time=12:00:00             # 12 hours (sequential may take longer)
#SBATCH --partition=shared          # or serial, test, etc.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # Single-threaded
#SBATCH --mem=16G                   # 16GB RAM
#SBATCH --mail-type=END,FAIL        # Email when done or failed
#SBATCH --mail-user=truongtruong@fas.harvard.edu  

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Starting at: $(date)"

module load julia/1.11.5

cd $SLURM_SUBMIT_DIR

echo "Julia version:"
julia --version
echo "Project directory: $(pwd)"

# Run the inference
julia --project=. inference_multi.jl

echo "Finished at: $(date)"

