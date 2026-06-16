#!/bin/bash
#SBATCH --job-name=probe_langs
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=probe_langs_%j.out
#SBATCH --error=probe_langs_%j.err

# Load environment
module purge
module load cuda

source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
conda activate nllb

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SCRIPT_DIR=/home/common/ACNLP/eng2nav/nllb_ft/lang_codes

python $SCRIPT_DIR/probe_nllb_langs.py \
    --eng_file /home/common/ACNLP/eng2nav/nllb_ft/my_english.txt \
    --nav_file /home/common/ACNLP/eng2nav/nllb_ft/my_navajo.txt \
    --n_sentences 500 \
    --model_size 600M \
    --output_csv $SCRIPT_DIR/probe_results_NEW.csv
