#!/bin/bash
#SBATCH --job-name=mdlm-owt-scratch
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --error=watch_folder/%x_%j.err
# If `sinfo -o "%P %G %D %N" | grep bigbatch` shows a real Gres string
# (not "(null)"), uncomment and fill in below:
# #SBATCH --gres=gpu:<type>:1
# NOTE: bigbatch has 1 GPU/node (the authors' original template requested
# 4 GPUs/node) -- ntasks-per-node is set to 1 to match mscluster's hardware.

set -e

# --- Repo location ---
REPO_DIR=/home-mscluster/$USER/mdlm/mdlm-african-lang
cd "$REPO_DIR"

# Ensure output dirs exist (README convention: outputs/ for runs+checkpoints,
# watch_folder/ for slurm logs)
mkdir -p outputs watch_folder

# --- Environment ---
# Non-interactive batch shells do NOT source ~/.bashrc automatically,
# so conda must be activated explicitly here rather than relying on it.
# (Deliberately not using --get-user-env: it's known to behave
# inconsistently with conda's shell-function-based activate.)
source /home-mscluster/$USER/miniconda3/etc/profile.d/conda.sh
conda activate mdlm

# NOTE: deliberately NOT exporting the system CUDA 12.9 PATH/LD_LIBRARY_PATH
# here. That was only needed to build flash-attn against nvcc. At runtime,
# torch/flash-attn use the CUDA runtime bundled inside the conda env
# (pytorch-cuda=12.1); mixing in the system CUDA libs at runtime could
# cause version conflicts rather than help anything.

export HYDRA_FULL_ERROR=1

# --- Training ---
# Fixed run directory (not the default timestamped folder) so that a
# requeued/resubmitted job resumes from the same checkpoint directory
# instead of starting a fresh run at step 0.
#
# The sibling kuleshov-group repo's own script comment says to set
# EITHER `hydra.run.dir` OR `checkpointing.save_dir` for this to work.
# hydra.run.dir is confirmed to exist (used successfully in this repo
# already for inference). checkpointing.save_dir is NOT yet confirmed
# for this specific repo's config schema -- run:
#     grep -rn "save_dir" configs/
# before relying on it. If it exists as its own independent key (not
# something that already interpolates from hydra.run.dir), uncomment
# the line below and adjust the path accordingly. If you add it and it
# is not a real config key, Hydra will error on an unknown override.
RUN_DIR="$REPO_DIR/outputs/owt_scratch"

srun python -u main.py \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=small \
  data=openwebtext-split \
  data.cache_dir=/datasets/$USER/mdlm/owt_cache \
  wandb.name=mdlm-owt-scratch \
  wandb.mode=offline \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  hydra.run.dir="$RUN_DIR"
  # checkpointing.save_dir="$RUN_DIR/checkpoints" \   # uncomment once confirmed via grep above