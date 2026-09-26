#!/bin/bash
#SBATCH --job-name=ppl-swahili
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=02:00:00
#SBATCH --exclude=mscluster[44-48,50,51,57,59,62,64-70,74-76,78,83,85]
#SBATCH --output=/home-mscluster/jtshibumbu/mdlm/mdlm-african-lang/watch_folder/%x_%j.out
#SBATCH --error=/home-mscluster/jtshibumbu/mdlm/mdlm-african-lang/watch_folder/%x_%j.out

# Usage: sbatch ppl_eval_swahili.sh <cell> [checkpoint]
#   cell: ar_unigram | ar_bpe | mdlm_unigram | mdlm_bpe
set -e
REPO_DIR=/home-mscluster/$USER/mdlm/mdlm-african-lang
cd "$REPO_DIR"
source /home-mscluster/$USER/miniconda3/etc/profile.d/conda.sh
conda activate mdlm
export HYDRA_FULL_ERROR=1

CELL=${1:?give a cell: ar_unigram | ar_bpe | mdlm_unigram | mdlm_bpe}
case $CELL in
  ar_unigram)   RUN=outputs/swahili_ar;       DATA=swahili;     MODEL="model=small-ar parameterization=ar backbone=ar" ;;
  ar_bpe)       RUN=outputs/swahili_ar_bpe;   DATA=swahili-bpe; MODEL="model=small-ar parameterization=ar backbone=ar" ;;
  mdlm_unigram) RUN=outputs/swahili_scratch;  DATA=swahili;     MODEL="model=small parameterization=subs backbone=dit" ;;
  mdlm_bpe)     RUN=outputs/swahili_mdlm_bpe; DATA=swahili-bpe; MODEL="model=small parameterization=subs backbone=dit" ;;
  *) echo "unknown cell $CELL" >&2; exit 1 ;;
esac

# Default checkpoint: step 7500 (the matched point for the 2x2), else last.ckpt
CKPT=${2:-$(find "$RUN/checkpoints" -name "*step=7500*" | head -1)}
CKPT=${CKPT:-$RUN/checkpoints/last.ckpt}
echo "host=$(hostname)  cell=$CELL  checkpoint=$CKPT"

python - << 'PYEOF' || { echo "ERROR: no usable GPU on $(hostname)" >&2; exit 1; }
import sys, torch
sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PYEOF

python -u main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=$DATA \
  $MODEL \
  model.length=1024 \
  eval.checkpoint_path="$REPO_DIR/$CKPT" \
  wandb.name=ppl-swahili-$CELL \
  +wandb.offline=true \
  hydra.run.dir="$REPO_DIR/outputs/ppl_eval_$CELL"