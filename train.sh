#!/usr/bin/bash
#SBATCH --job-name=train_cellot
#SBATCH --output=train_cellot.%j.out
#SBATCH --error=train_cellot.%j.err
#SBATCH --time=60:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8GB

module load python/3.9.0

source cellot_sherlock_venv/bin/activate

python ./scripts/train.py \
  --outdir ./results/sherlock/test_1/model-cellot \
  --config ./configs/tasks/sherlock.yaml \
  --config ./configs/models/cellot.yaml \
  --config.data.target LPS
