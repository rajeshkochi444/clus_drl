#!/bin/bash
#SBATCH --job-name=Wang
#SBATCH --ntasks=1
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu 8G


module purge
module load bluebear


cd "$PBS_O_WORKDIR"


source "/rds/projects/2018/johnston-copper-clusters-rr/Rajesh-2/Anaconda3/etc/profile.d/conda.sh"
conda activate catgym
python gym_trpo_single_training.py
