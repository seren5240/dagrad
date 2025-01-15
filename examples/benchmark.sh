#!/bin/bash
#
#SBATCH --account=pi-naragam
#SBATCH --job-name=golem_benchmark
#SBATCH --output=./slurm/out/%j.%N.stdout
#SBATCH --error=./slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/skwok1/dagrad/examples
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=900
#SBATCH --exclusive
#SBATCH --time=2-00:00:00

set -e

cd ..
pip3 install -e .
cd examples

echo "Running benchmark"

python3 -u "./benchmark.py" || exit 1
