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
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2000
#SBATCH --time=3-00:00:00

set -e

module load python/booth/3.12

cd ..
pip3 install -e .
cd examples

echo "Running benchmark"

# python3 -u "./benchmark.py" 100 1000 0.5 gauss eq
# python3 -u "./benchmark.py" 100 1000 0.5 exp eq
# python3 -u "./benchmark.py" 100 1000 0.5 gumbel eq
# python3 -u "./benchmark.py" 100 1000 1 gauss eq
# python3 -u "./benchmark.py" 100 1000 1 exp eq
# python3 -u "./benchmark.py" 100 1000 1 gumbel eq
# python3 -u "./benchmark.py" 100 1000 2 gauss eq
# python3 -u "./benchmark.py" 100 1000 2 exp eq
# python3 -u "./benchmark.py" 100 1000 2 gumbel eq
python3 -u "./benchmark.py"

wait
