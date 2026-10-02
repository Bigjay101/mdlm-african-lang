#!/bin/bash
#SBATCH --job-name=bleu-swahili
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=watch_folder/bleu-swahili_%j.out
#SBATCH --exclude=mscluster[44-48,50,51,57,59,62,64-70,74-76,78,83,85]

# Usage (from the repo root):
#   sbatch bleu_eval.sh ar_unigram
#   sbatch bleu_eval.sh mdlm_unigram
#   sbatch bleu_eval.sh ar_bpe
#   sbatch bleu_eval.sh mdlm_bpe
# Optional 2nd argument: an explicit checkpoint path (default: that run's best.ckpt).
# When all four have finished:  python bleu_eval.py --summarise

set -euo pipefail
cd /home-mscluster/jtshibumbu/mdlm/mdlm-african-lang
mkdir -p watch_folder
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mdlm

CELL=${1:?give a cell: mdlm_unigram | ar_unigram | mdlm_bpe | ar_bpe}
case $CELL in
  mdlm_unigram) RUN=swahili_scratch;  ARGS="model=small parameterization=subs backbone=dit data=swahili" ;;
  ar_unigram)   RUN=swahili_ar;       ARGS="model=small-ar parameterization=ar backbone=ar data=swahili" ;;
  mdlm_bpe)     RUN=swahili_mdlm_bpe; ARGS="model=small parameterization=subs backbone=dit data=swahili-bpe" ;;
  ar_bpe)       RUN=swahili_ar_bpe;   ARGS="model=small-ar parameterization=ar backbone=ar data=swahili-bpe" ;;
  *) echo "unknown cell $CELL"; exit 1 ;;
esac
CKPT=${2:-outputs/$RUN/checkpoints/best.ckpt}

nvidia-smi
python bleu_eval.py --cell "$CELL" --ckpt "$CKPT" \
  -- $ARGS model.length=1024