#!/bin/bash
#SBATCH --job-name=mdlm-swahili-scratch
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --exclude=mscluster46
#SBATCH --error=watch_folder/%x_%j.err
# NOTE: --output/--error are relative, so this must be submitted from
# $REPO_DIR (as the working English run was). sbatch from elsewhere fails
# before the script body runs.

set -e

# --- Repo location ---
REPO_DIR=/home-mscluster/$USER/mdlm/mdlm-african-lang
cd "$REPO_DIR"

mkdir -p outputs watch_folder

# --- Environment ---
# Non-interactive batch shells do NOT source ~/.bashrc automatically,
# so conda must be activated explicitly here.
source /home-mscluster/$USER/miniconda3/etc/profile.d/conda.sh
conda activate mdlm

# NOTE: deliberately NOT exporting system CUDA PATH/LD_LIBRARY_PATH here --
# that was only needed to build flash-attn against nvcc. At runtime torch
# uses the CUDA runtime bundled in the conda env (pytorch-cuda=12.1).

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Sanity checks: fail fast rather than 20 minutes into a job ---
# Corpus must be extracted from Kaggle to the path dataloader.py expects.
N_DOCS=$(find configs/data/maneno-yetu/data-raw/cleaned -name "*.txt" 2>/dev/null | wc -l)
if [ "$N_DOCS" -ne 738 ]; then
  echo "ERROR: expected 738 corpus files, found $N_DOCS" >&2
  echo "  in $REPO_DIR/configs/data/maneno-yetu/data-raw/cleaned" >&2
  exit 1
fi
test -f configs/data/maneno-yetu/splits/split_manifest.json || \
  { echo "ERROR: split_manifest.json missing" >&2; exit 1; }
test -f configs/data/maneno-yetu/tokenizer/tokenizer.json || \
  { echo "ERROR: tokenizer missing" >&2; exit 1; }

# --- Training ---
# Fixed run dir so a requeued job resumes from the same checkpoints instead
# of a fresh timestamped folder at step 0.
# checkpointing.save_dir needs NO separate override: it resolves via the
# ${cwd:} resolver registered in main.py, and hydra.job.chdir=true means
# cwd is already RUN_DIR by the time it's read. (Confirmed in main.py.)
RUN_DIR="$REPO_DIR/outputs/swahili_scratch"

python -u main.py \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  model=small \
  data=swahili \
  data.cache_dir=/datasets/$USER/mdlm/swahili_cache \
  wandb.name=mdlm-swahili-scratch \
  +wandb.offline=true \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=False \
  sampling.steps=1000 \
  trainer.max_steps=10000 \
  trainer.val_check_interval=500 \
  hydra.run.dir="$RUN_DIR"