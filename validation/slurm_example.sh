#!/bin/bash -l
# Example Slurm array job for the full study: 36 scenarios x 50 seeds = 1800 runs,
# 5 runs per array element on one core each. Adapt partition, time and environment
# to your cluster, then from the repository root:
#     mkdir -p logs && sbatch validation/slurm_example.sh
#     python validation/summarise.py             # when all runs are done
#SBATCH --job-name=s2c-validation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --array=0-359
#SBATCH --output=logs/validation_%A_%a.out

# activate the Python environment that has struct2clust installed, e.g.
# conda activate struct2clust
python validation/run_study.py --seeds 50 --index "$SLURM_ARRAY_TASK_ID" --chunk 5
