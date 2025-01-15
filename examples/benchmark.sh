#!/bin/bash
#
#SBATCH --mail-user=serenkwok@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=golem_benchmark
#SBATCH --output=./slurm/out/%j.%N.stdout
#SBATCH --error=./slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/skwok1/dagrad/examples
#SBATCH --partition=gpu_h100
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=900
#SBATCH --exclusive
#SBATCH --time=2-00:00:00

set -e

python3 -u "./benchmark.py" || exit 1
