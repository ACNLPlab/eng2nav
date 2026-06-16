#!/bin/bash
#SBATCH --job-name=test_nsl_nllb
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=test_files/grn_NSL_600M_%j.out
#SBATCH --error=test_files/grn_NSL_600M_%j.err

module purge
source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
conda activate nllb
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
pip install --quiet protobuf sentencepiece

echo "=========================================="
echo "Testing NSL English→Navajo NLLB200 Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

BASE_DIR=/home/common/ACNLP/eng2nav/NSL_nllb_ft
FOLD_DIR=$BASE_DIR/grn_NSL_600M/fold_4

python test_nsl_nllb.py \
    --checkpoint_dir $FOLD_DIR/checkpoint-75870 \
    --data_dir       $FOLD_DIR/data \
    --src_lang       eng_Latn \
    --tgt_lang       grn_Latn \
    --batch_size     32 \
    --max_length     256 \
    --output_dir     $FOLD_DIR/results

if [ $? -ne 0 ]; then
    echo "ERROR: Translation failed. Check the .err log."
    exit 1
fi

echo "=========================================="
echo "Testing complete!"
echo "Predictions saved to: $FOLD_DIR/results/test_predictions.txt"
echo "=========================================="
