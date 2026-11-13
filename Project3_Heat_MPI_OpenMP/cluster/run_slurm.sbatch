#!/bin/bash
#SBATCH --job-name=jacobi_hybrid
#SBATCH --output=jacobi_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=8G

# Alternative cluster launcher ONLY if your site uses Slurm.
# If a partition/account is required, add --partition and/or --account.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# module load gcc
# module load openmpi
command -v mpicxx >/dev/null || { echo 'ERROR: mpicxx unavailable; load MPI module'; exit 1; }
command -v mpiexec >/dev/null || { echo 'ERROR: mpiexec unavailable; load MPI module'; exit 1; }

make -j 2
mkdir -p results
python3 scripts/validate.py
python3 scripts/benchmark.py --scaling strong --nx 1024 --ny 1024 \
    --steps 400 --ranks 1,2,4 --threads 1,2 --max-cores 8 \
    --repeats 3 --warmup 1 --output results/strong_scaling.csv
python3 scripts/benchmark.py --scaling weak --nx 512 --ny 512 \
    --steps 300 --ranks 1,2,4 --threads 1,2 --max-cores 8 \
    --repeats 3 --warmup 1 --output results/weak_scaling.csv
