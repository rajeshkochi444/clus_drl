#!/bin/bash
#SBATCH --job-name=drl
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu 8G
#SBATCH --partition=l_long
#SBATCH --qos=ll


cd "$PBS_O_WORKDIR"

echo "Starting at "`date`
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# Source conda.sh to make conda executables available in the environment
source "/ddn/home/raraju042/anaconda3/etc/profile.d/conda.sh" 

# Activate Conda environment which you created and tested earlier
conda activate clusgym_dscribe

# Launch computation
#srun --ntasks=1 python3 DLcode.py
python gym_trpo_parallel_training.py


echo "Simulation Ending.."
echo "Ending at "`date`

#alias ffmpeg="ffmpeg -threads 0"
#srun -n 1 -c 16 python gym_trpo_parallel_training.py
#srun -n 16 python gym_trpo_parallel_training.py
#python gym_trpo_parallel_training.py
