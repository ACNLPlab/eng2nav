#!/bin/bash
#SBATCH --job-name=train_nllb
#SBATCH --partition=gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=out_files/%j_%x.out
#SBATCH --error=out_files/%j_%x.err
# ============================================================================
# change these vars
# ============================================================================
# EXPERIMENT NAME (affects output directory names)
EXPERIMENT_NAME="fin_NOsyll_600M"
# LANGUAGE SETTINGS
TARGET_LANG="fin_Latn"           # NLLB language code
SYLLABIFY="--no_syllabify"       # Use "--syllabify" or "--no_syllabify"
# MODEL SETTINGS
MODEL_SIZE="600M"                 # "600M" or "3.3B"
LEARNING_RATE="2e-5"              # 2e-5 for 600M, 1e-5 for 3.3B
# TRAINING SETTINGS
EPOCHS="10"
BATCH_SIZE="4"
GRAD_ACCUM="4"
MAX_LENGTH="256"
# FOLD SETTINGS
# Use "all" to run all folds, or a single integer (0,1,2,3,4) to re-run
# one specific fold using the data splits already saved on disk under
# <EXPERIMENT_NAME>/fold_<N>/data/. The raw input files are not needed
# when re-running a single fold (they are ignored).
FOLD_NUM="all"
# FILE PATHS (change if your files are in different locations)
BIBLE_ENG="aligned_bibles/english_nlt.txt"
BIBLE_NAV="aligned_bibles/navajo_nvjob.txt"
USER_ENG="my_english.txt"
USER_NAV="my_navajo.txt"
REPO_PATH="Low_Resource_MT"

# ============================================================================
# do not change rest :)
# ============================================================================

echo "========================================================"
echo "STARTING EXPERIMENT: $EXPERIMENT_NAME"
echo "Started at: $(date)"
echo "========================================================"
# Load environment
module purge
module load cuda/12.1
source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
conda activate nllb
# Set environment variables
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Create directories
SCRIPT_DIR=/home/common/ACNLP/eng2nav/nllb_ft
mkdir -p $SCRIPT_DIR/out_files
mkdir -p $SCRIPT_DIR/$EXPERIMENT_NAME
# Show configuration
echo "CONFIGURATION:"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Target language: $TARGET_LANG"
echo "  Syllabification: $SYLLABIFY"
echo "  Model: $MODEL_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Fold: $FOLD_NUM"
echo "  Output dir: $EXPERIMENT_NAME"
echo "========================================================"

# When re-running a single fold, the saved data splits on disk are used
# and the raw input files are not read — but we still pass them so the
# argument parser doesn't complain. File-existence checks are skipped for
# single-fold mode since the data was already validated in the original run.
if [ "$FOLD_NUM" = "all" ]; then
    # Full run: verify input files exist
    for file in "$BIBLE_ENG" "$BIBLE_NAV" "$USER_ENG" "$USER_NAV"; do
        if [ ! -f "$SCRIPT_DIR/$file" ]; then
            echo "ERROR: File not found: $SCRIPT_DIR/$file"
            exit 1
        fi
    done
    echo "All input files found."
else
    # Single-fold re-run: verify the saved splits exist
    FOLD_DATA_DIR="$SCRIPT_DIR/$EXPERIMENT_NAME/fold_${FOLD_NUM}/data"
    echo "Single-fold mode: checking for saved splits in $FOLD_DATA_DIR"
    for split_file in "train.eng" "train.nav" "dev.eng" "dev.nav" "test.eng" "test.nav"; do
        if [ ! -f "$FOLD_DATA_DIR/$split_file" ]; then
            echo "ERROR: Saved split file not found: $FOLD_DATA_DIR/$split_file"
            echo "Make sure you have completed a full run first to generate data splits."
            exit 1
        fi
    done
    echo "All saved split files found for fold $FOLD_NUM."
fi
echo "========================================================"

# Show GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run training
echo "Starting training at: $(date)"
python train_nllb.py \
    --model_size $MODEL_SIZE \
    --tgt_lang $TARGET_LANG \
    $SYLLABIFY \
    --repo_path $REPO_PATH \
    --bible_eng_file $BIBLE_ENG \
    --bible_nav_file $BIBLE_NAV \
    --user_eng_file $USER_ENG \
    --user_nav_file $USER_NAV \
    --output_dir $EXPERIMENT_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --fold_num $FOLD_NUM

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo "========================================================"
    echo "TRAINING COMPLETED SUCCESSFULLY at: $(date)"
    echo "Results saved in: $SCRIPT_DIR/$EXPERIMENT_NAME"
    # Show best result if available
    if [ -f "$SCRIPT_DIR/$EXPERIMENT_NAME/cv_results.json" ]; then
        BEST_FOLD=$(grep -o '"best_fold": [0-9]*' $SCRIPT_DIR/$EXPERIMENT_NAME/cv_results.json | cut -d' ' -f2)
        BEST_LOSS=$(grep -o '"best_eval_loss": [0-9.]*' $SCRIPT_DIR/$EXPERIMENT_NAME/cv_results.json | cut -d' ' -f2)
        echo "Best fold: $BEST_FOLD, Best loss: $BEST_LOSS"
    fi
else
    echo "========================================================"
    echo "TRAINING FAILED at: $(date)"
    exit 1
fi
echo "========================================================"
