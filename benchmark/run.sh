#!/bin/bash
#
#SBATCH --account=pi-naragam
#SBATCH --job-name=flex_benchmark
#SBATCH --output=./slurm/out/%j.%N.stdout
#SBATCH --error=./slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/skwok1/dagrad/benchmark
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2000
#SBATCH --time=3-00:00:00
#SBATCH --exclusive

set -e

module load python/booth/3.12
module load R/4.3/4.3.2

cd ..
pip3 install -e .
pip3 install matplotlib
cd benchmark

# echo "Installing R packages"

# Rscript -e 'install.packages(c("V8"),lib="~/Rlibs", repos="http://cran.us.r-project.org", quiet=TRUE, verbose=FALSE)'
# Rscript -e 'install.packages(c("sfsmisc"), lib="~/Rlibs", repos="http://cran.us.r-project.org", quiet=TRUE, verbose=FALSE)'
# Rscript -e 'install.packages(c("clue"),lib="~/Rlibs", repos="http://cran.us.r-project.org", quiet=TRUE, verbose=FALSE)'
# Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/randomForest/randomForest_4.6-14.tar.gz", lib="~/Rlibs", repos=NULL, type="source")'
# Rscript -e 'install.packages(c("lattice"),lib="~/Rlibs",repos="http://cran.us.r-project.org", quiet=TRUE, verbose=FALSE)'
# Rscript -e 'install.packages(c("devtools"),lib="~/Rlibs",repos="http://cran.us.r-project.org", quiet=TRUE, verbose=FALSE)'
# Rscript -e 'install.packages(c("MASS"),lib="~/Rlibs",repos="http://cran.us.r-project.org", quiet=TRUE, verbose=FALSE)'
# Rscript -e 'install.packages("BiocManager",lib="~/Rlibs", repos="http://cran.us.r-project.org")'
# Rscript -e '.libPaths("~/Rlibs"); BiocManager::install(c("igraph"),lib="~/Rlibs")'
# Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/fastICA/fastICA_1.2-2.tar.gz",lib="~/Rlibs", repos=NULL, type="source")'
# Rscript -e '.libPaths("~/Rlibs"); BiocManager::install(c("SID", "bnlearn", "pcalg", "kpcalg", "glmnet", "mboost"))'
# Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/CAM/CAM_1.0.tar.gz",lib="~/Rlibs", repos=NULL, type="source")'
# Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/sparsebnUtils_0.0.8.tar.gz",lib="~/Rlibs", repos=NULL, type="source")'
# Rscript -e '.libPaths("~/Rlibs"); BiocManager::install(c("ccdrAlgorithm", "discretecdAlgorithm"),lib="~/Rlibs")'

# Rscript -e 'install.packages("devtools",lib="~/Rlibs", repos="http://cran.us.r-project.org")'
# Rscript -e '.libPaths("~/Rlibs"); library(devtools); install_github("cran/CAM"); install_github("cran/momentchi2"); install_github("Diviyan-Kalainathan/RCIT", quiet=TRUE, verbose=FALSE)'
# Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/sparsebn/sparsebn_0.1.2.tar.gz",lib="~/Rlibs", repos=NULL, type="source")'

# Rscript -e '
# .libPaths("~/Rlibs"); BiocManager::install(c("RBGL", "graph"), ask=FALSE);
# install.packages("SID", lib="~/Rlibs", repos="https://mirror.las.iastate.edu/CRAN/", dependencies=TRUE)
# '
# Rscript -e '.libPaths("~/Rlibs"); library(SID); print("SID package loaded successfully")'

echo "Running benchmark"

python3 ./runner.py
