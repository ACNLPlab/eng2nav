#!/bin/bash
#SBATCH --job-name=test_nsl_m2m
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=test_files/sw_NSL_418M_fold0.out
#SBATCH --error=test_files/sw_NSL_418M_fold0.err

module purge
source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
conda activate nllb
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
pip install --quiet protobuf sentencepiece

echo "=========================================="
echo "Testing NSL English→Navajo M2M100 Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

BASE_DIR=/home/common/ACNLP/eng2nav/NSL_m2m_ft
FOLD_DIR=$BASE_DIR/sw_NSL_418M/fold_4

python test_nsl_m2m.py \
    --checkpoint_dir $FOLD_DIR/checkpoint-45522 \
    --data_dir       $FOLD_DIR/data \
    --src_lang       en \
    --tgt_lang       sw \
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
