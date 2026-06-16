#!/bin/bash
#SBATCH --job-name=test_m2m
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=test_files/fi_NOsyll_fold0.out
#SBATCH --error=test_files/fi_NOsyll_fold0.err

module purge
module load cuda/12.1

source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
conda activate nllb

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip install --quiet protobuf sentencepiece

echo "=========================================="
echo "Testing English→Navajo M2M100 Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

BASE_DIR=/home/common/ACNLP/eng2nav/m2m_ft

# M2M100: Use best checkpoint from fold
python test_m2m.py \
    --checkpoint_dir $BASE_DIR/fi_NOsyll_418M/fold_0/checkpoint-15970 \
    --data_dir $BASE_DIR/fi_NOsyll_418M/fold_0/data \
    --src_lang en \
    --tgt_lang fi \
    --batch_size 32 \
    --output_file $BASE_DIR/fi_NOsyll_418M/fold_0/test_preds.txt

echo "=========================================="
echo "Testing complete!"
echo "Predictions saved to: test_preds.txt"
echo "=========================================="
