#!/bin/bash
#SBATCH --job-name=train_nsl_m2m
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=out_files/%j_%x.out
#SBATCH --error=out_files/%j_%x.err

# ============================================================================
# NSL-MT — change these variables
# ============================================================================

EXPERIMENT_NAME="grn_NSL_600M"

TARGET_LANG="grn_Latn"
MODEL_SIZE="600M"             # "418M" or "1.2B"
LEARNING_RATE="2e-5"
EPOCHS="10"
BATCH_SIZE="4"
GRAD_ACCUM="4"
MAX_LENGTH="256"
FOLD_NUM="all"                # "all" or 0-4

# Pool A: standard parallel data (bible + user) — goes through k-fold
BIBLE_ENG="aligned_bibles/english_nlt.txt"
BIBLE_NAV="aligned_bibles/navajo_nvjob.txt"
USER_ENG="my_english.txt"
USER_NAV="my_navajo.txt"

# Pool B: full morphosyntactic gloss file — split and violated per fold
# Do NOT pre-split this file. Pass the complete dataset here.
NSL_GLOSS="xml_data.txt"

# NSL loss weight α 
NSL_ALPHA="0.7"

REPO_PATH="Low_Resource_MT"

# ============================================================================
# Do not change below this line
# ============================================================================

echo "========================================================"
echo "NSL-MT DUAL K-FOLD: $EXPERIMENT_NAME"
echo "Started: $(date)"
echo "========================================================"

module purge
module load cuda/12.1
source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
conda activate nllb

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR=/home/common/ACNLP/eng2nav/NSL_nllb_ft
mkdir -p $SCRIPT_DIR/out_files
mkdir -p $SCRIPT_DIR/$EXPERIMENT_NAME

echo "CONFIGURATION:"
echo "  Experiment:  $EXPERIMENT_NAME"
echo "  Model:       NLLB $MODEL_SIZE"
echo "  Target lang: $TARGET_LANG"
echo "  NSL alpha:   $NSL_ALPHA"
echo "  Gloss file:  $NSL_GLOSS"
echo "  Fold:        $FOLD_NUM"
echo "========================================================"

# Verify gloss file exists
if [ ! -f "$SCRIPT_DIR/$NSL_GLOSS" ]; then
    echo "ERROR: Gloss file not found: $SCRIPT_DIR/$NSL_GLOSS"
    exit 1
fi
echo "Gloss file verified: $NSL_GLOSS"

# Verify Pool A files
if [ "$FOLD_NUM" = "all" ]; then
    for file in "$BIBLE_ENG" "$BIBLE_NAV" "$USER_ENG" "$USER_NAV"; do
        if [ ! -f "$SCRIPT_DIR/$file" ]; then
            echo "ERROR: Pool A file not found: $SCRIPT_DIR/$file"
            exit 1
        fi
    done
    echo "Pool A files verified."
fi

# Also verify navajo_violation_generators.py is present
if [ ! -f "$SCRIPT_DIR/navajo_violation_generators.py" ]; then
    echo "ERROR: navajo_violation_generators.py not found in $SCRIPT_DIR"
    echo "This file must be in the same directory as train_nsl_m2m.py"
    exit 1
fi
echo "Violation generators verified."
echo "========================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Starting training: $(date)"

python train_nsl_nllb.py \
    --model_size      $MODEL_SIZE \
    --tgt_lang        $TARGET_LANG \
    --bible_eng_file  $BIBLE_ENG \
    --bible_nav_file  $BIBLE_NAV \
    --user_eng_file   $USER_ENG \
    --user_nav_file   $USER_NAV \
    --nsl_gloss_file  $NSL_GLOSS \
    --nsl_alpha       $NSL_ALPHA \
    --output_dir      $EXPERIMENT_NAME \
    --epochs          $EPOCHS \
    --batch_size      $BATCH_SIZE \
    --grad_accum      $GRAD_ACCUM \
    --learning_rate   $LEARNING_RATE \
    --max_length      $MAX_LENGTH \
    --fold_num        $FOLD_NUM

if [ $? -eq 0 ]; then
    echo "========================================================"
    echo "TRAINING COMPLETE: $(date)"
    if [ -f "$SCRIPT_DIR/$EXPERIMENT_NAME/cv_results.json" ]; then
        BEST_FOLD=$(grep -o '"best_fold": [0-9]*' \
            $SCRIPT_DIR/$EXPERIMENT_NAME/cv_results.json | cut -d' ' -f2)
        BEST_LOSS=$(grep -o '"best_eval_loss": [0-9.]*' \
            $SCRIPT_DIR/$EXPERIMENT_NAME/cv_results.json | cut -d' ' -f2)
        echo "Best fold: $BEST_FOLD  |  eval_loss: $BEST_LOSS"
    fi
else
    echo "TRAINING FAILED: $(date)"
    exit 1
fi
echo "========================================================"
