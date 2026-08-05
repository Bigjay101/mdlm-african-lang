#!/bin/bash
#SBATCH --job-name=mdlm-owt-sampleeval
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=02:00:00
#SBATCH --output=/home-mscluster/CHANGE_ME/mdlm/mdlm-african-lang/watch_folder/%x_%j.out
#SBATCH --error=/home-mscluster/CHANGE_ME/mdlm/mdlm-african-lang/watch_folder/%x_%j.err

# If `sinfo -o "%P %G %D %N" | grep bigbatch` shows a real Gres string
# (not "(null)"), uncomment and fill in:
# #SBATCH --gres=gpu:<type>:1
# bigbatch has 1 GPU/node; ntasks-per-node=1 matches that.

set -e
set -o pipefail   # so a python failure still aborts even though we pipe to tee

# --- Repo location ---
REPO_DIR=/home-mscluster/$USER/mdlm/mdlm-african-lang
cd "$REPO_DIR"
mkdir -p watch_folder

# --- The checkpoint to sample from ---
# Point at the .ckpt FILE, not the checkpoints/ directory. Confirm with `ls` first.
CKPT="$REPO_DIR/outputs/owt_scratch_excl/checkpoints/last.ckpt"

# --- Dedicated results file (generated text + generative perplexity) ---
RESULTS="$REPO_DIR/watch_folder/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_results.txt"

# --- Caches ---
# data.cache_dir overrides the authors' hardcoded /share/kuleshov path.
# HF_HOME holds the GPT-2 model used to score generative perplexity of the samples.
DATA_CACHE=/gluster/$USER/mdlm_data_cache
export HF_HOME=/gluster/$USER/hf_home
mkdir -p "$DATA_CACHE" "$HF_HOME"

# --- AIRGAP: sample_eval downloads GPT-2 (for generative perplexity) from HF.
# If bigbatch nodes have no internet, fetch it once on the login node into
# HF_HOME, then uncomment: ---
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# --- Environment ---
source /home-mscluster/$USER/miniconda3/etc/profile.d/conda.sh
conda activate mdlm

# NOT exporting system CUDA at runtime -- torch/flash-attn use the CUDA runtime
# bundled in the conda env. System CUDA was only for building flash-attn.

export HYDRA_FULL_ERROR=1

echo "=== Job ${SLURM_JOB_ID} on $(hostname) $(date) ==="
nvidia-smi
ls -lh "$CKPT"

# --- Preflight: fail fast on an env/ABI problem instead of after queueing ---
python -c "import torch, flash_attn; print('torch', torch.__version__, '| flash_attn', flash_attn.__version__, '| cuda:', torch.cuda.is_available())"

# --- Sample generation + generative perplexity ---
# generate_samples() prints 'Text samples:' and 'Generative perplexity:' to
# stdout, so tee captures both into $RESULTS.
# backbone=dit + a local .ckpt (use backbone=hf_dit only for the HF-hosted model).
# model/parameterization/length MUST match how the checkpoint was trained.
# Start small: num_sample_batches=2. Raise it once you're happy with the output.
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=/path/to/checkpoint/mdlm.ckpt \
  data=openwebtext-split  \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=10000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=1 \
  backbone=dit

# Pull the generated text + generative perplexity into a summary block.
# Read into a variable FIRST, then append -- don't grep a file while writing it.
SUMMARY=$(grep -iE "Text samples:|Generative perplexity:|gen_ppl" "$RESULTS" || true)
{ echo; echo "----- summary -----"; echo "$SUMMARY"; } >> "$RESULTS"

echo "=== Job ${SLURM_JOB_ID} finished $(date) ==="
echo "Samples + generative perplexity written to: $RESULTS"