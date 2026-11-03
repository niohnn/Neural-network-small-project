#!/bin/bash
#SBATCH -J parprog
#SBATCH -e /work/scratch/kurse/kurs00107/<TUID>/stderr/stderr.parprog.%j.txt
#SBATCH -o /work/scratch/kurse/kurs00107/<TUID>/stdout/stdout.parprog.%j.txt
#SBATCH -C avx512
#SBATCH -n 1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=60
#SBATCH --cpus-per-task=1
#SBATCH -c 1
#SBATCH -A kurs00107
#SBATCH -p kurs00107
#SBATCH --reservation=kurs00107
module load gcc/13 cmake/4.2
echo "This is job $SLURM_JOB_ID"
build/DNN_app
