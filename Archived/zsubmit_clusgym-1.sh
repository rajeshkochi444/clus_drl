#!/bin/bash
#SBATCH --job-name=Wang
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu 8G


module purge
module load bluebear

module load bear-apps/2022a
module load FFmpeg/4.4.2-GCCcore-11.3.0

cd "$PBS_O_WORKDIR"

export OMP_NUM_THREADS=1

source "/rds/projects/2018/johnston-copper-clusters-rr/Rajesh-2/Anaconda3/etc/profile.d/conda.sh"
conda activate catgym

#alias ffmpeg="ffmpeg -threads 0"
#srun -n 1 -c 16 python gym_trpo_parallel_training.py
#srun -n 16 python gym_trpo_parallel_training.py
python gym_trpo_parallel_training.py
