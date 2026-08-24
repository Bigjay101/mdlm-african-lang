#!/bin/bash
#SBATCH --job-name=mdlm-swahili-sample
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=/home-mscluster/jtshibumbu/mdlm/mdlm-african-lang/watch_folder/%x_%j.out
#SBATCH --error=/home-mscluster/jtshibumbu/mdlm/mdlm-african-lang/watch_folder/%x_%j.err

set -e
cd /home-mscluster/$USER/mdlm/mdlm-african-lang
source /home-mscluster/$USER/miniconda3/etc/profile.d/conda.sh
conda activate mdlm
export HYDRA_FULL_ERROR=1

python -u main.py \
  mode=sample_eval \
  eval.checkpoint_path=/home-mscluster/$USER/mdlm/mdlm-african-lang/outputs/swahili_scratch/checkpoints/last.ckpt \
  data=swahili \
  model.length=1024 \
  sampling.predictor=ddpm_cache \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=1 \
  backbone=dit \
  hydra.run.dir=/home-mscluster/$USER/mdlm/mdlm-african-lang/outputs/swahili_sample_eval

