#!/bin/bash
#SBATCH --job-name=test_nllb
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=test_files/fin_syll_600M_fold0.out
#SBATCH --error=test_files/fin_syll_600M_fold0.err

module purge
module load cuda/12.1
source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
conda activate nllb
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
pip install --quiet protobuf sentencepiece

echo "=========================================="
echo "Testing English→Navajo Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

BASE_DIR=/home/common/ACNLP/eng2nav/nllb_ft

python test_nllb.py \
    --checkpoint_dir $BASE_DIR/fin_syll_600M/fold_0/checkpoint-15970 \
    --data_dir $BASE_DIR/fin_syll_600M/fold_0/data \
    --lang_trg fin_Latn \
    --output_file $BASE_DIR/fin_syll_600M/fold_0/test_preds.txt

echo "=========================================="
echo "Testing complete!"
echo "=========================================="
