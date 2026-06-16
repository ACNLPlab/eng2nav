#!/usr/bin/env python3
"""
English→Navajo Training with Aligned Bible Data

FINAL RESEARCH VERSION (FIXED)
- 5-fold cross validation with proper fold splitting
- Each fold: train on 4 folds, validate on half of remaining fold
- Test portions (other half of each fold) saved separately for later evaluation
- Saved data splits per fold
- Step-based evaluation
- Cosine LR schedule
- Early stopping
- Safe decoder_input_ids handling (FIXED)
- Enhanced debugging
- Syllabification toggle (--syllabify / --no_syllabify)
- --fold_num: re-run a single fold using pre-saved data splits
"""

import argparse
import sys
import random
import json
from pathlib import Path
from typing import List, Optional
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

NLLB_MODELS = {
    "600M": "facebook/nllb-200-distilled-600M",
    "3.3B": "facebook/nllb-200-3.3B",
}

DEFAULT_LR = {
    "600M": 2e-5,
    "3.3B": 1e-5,
}


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize apostrophe variants to ASCII apostrophe (U+0027)."""
    return text.replace("\u2019", "'").replace("\u02bc", "'")


# ---------------------------------------------------------------------------
# FOMA syllabifier
# ---------------------------------------------------------------------------

class FOMASyllabifier:

    def __init__(self, repo_path: str):
        import os

        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            lib_path = os.path.join(conda_prefix, "lib")
            os.environ["LD_LIBRARY_PATH"] = (
                f"{lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            )

        self.repo_path = Path(repo_path)
        python_dir = self.repo_path / "python"

        if str(python_dir) not in sys.path:
            sys.path.insert(0, str(python_dir))

        import foma
        foma_bin = python_dir / "foma_tokenizer" / "nav_wordtokenizer.fomabin"
        print(f"\n[DEBUG] Loading FOMA binary from: {foma_bin}")
        self.fst = foma.FST.load(str(foma_bin))
        print("[DEBUG] FOMA loaded successfully")

    def syllabify_text(self, text: str) -> str:
        text = normalize_text(text)
        output = []

        for tok in text.split():
            try:
                result = self.fst[tok]
                if result:
                    out = result[0]
                    output.append(out.decode("utf-8") if isinstance(out, bytes) else out)
                else:
                    output.append(tok)
            except Exception as e:
                print(f"[DEBUG] FOMA error on token '{tok}': {e}")
                output.append(tok)

        return " ".join(output)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parallel_file(eng_file: str, nav_file: str, label: str):
    print(f"\n{'='*60}")
    print(f"[DEBUG] Loading {label} data...")
    print(f"[DEBUG] English file: {eng_file}")
    print(f"[DEBUG] Navajo file: {nav_file}")
    
    with open(eng_file, "r", encoding="utf-8") as f:
        eng_lines = [l.strip() for l in f if l.strip()]
    with open(nav_file, "r", encoding="utf-8") as f:
        nav_lines = [l.strip() for l in f if l.strip()]

    if len(eng_lines) != len(nav_lines):
        raise ValueError(f"{label} mismatch: {len(eng_lines)} vs {len(nav_lines)}")

    print(f"[DEBUG] Loaded {len(eng_lines):,} pairs")
    print(f"[DEBUG] First English sample: {eng_lines[0][:100]}...")
    print(f"[DEBUG] First Navajo sample: {nav_lines[0][:100]}...")
    print(f"{'='*60}")
    
    return eng_lines, nav_lines


# ---------------------------------------------------------------------------
# Load a single fold's data from pre-saved splits on disk
# ---------------------------------------------------------------------------

def load_fold_from_disk(fold_idx: int, output_dir: str):
    """
    Load train/dev/test splits for a single fold from previously saved data files.
    These are written by kfold_split() under <output_dir>/fold_<N>/data/.
    """
    data_dir = Path(output_dir) / f"fold_{fold_idx}" / "data"

    print(f"\n{'='*60}")
    print(f"[DEBUG] Loading fold {fold_idx} data from disk: {data_dir}")

    required = ["train.eng", "train.nav", "dev.eng", "dev.nav", "test.eng", "test.nav"]
    for fname in required:
        fpath = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Expected split file not found: {fpath}\n"
                f"Make sure you have already run a full kfold pass so that the "
                f"data splits are saved to disk, then re-run with --fold_num {fold_idx}."
            )

    def read_lines(path):
        return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    train_eng = read_lines(data_dir / "train.eng")
    train_nav = read_lines(data_dir / "train.nav")
    dev_eng   = read_lines(data_dir / "dev.eng")
    dev_nav   = read_lines(data_dir / "dev.nav")
    test_eng  = read_lines(data_dir / "test.eng")
    test_nav  = read_lines(data_dir / "test.nav")

    print(f"[DEBUG] Loaded fold {fold_idx} from disk:")
    print(f"  Train: {len(train_eng):,} examples")
    print(f"  Dev:   {len(dev_eng):,} examples")
    print(f"  Test:  {len(test_eng):,} examples (saved, not used in training)")
    print(f"{'='*60}")

    return train_eng, train_nav, dev_eng, dev_nav, test_eng, test_nav


# ---------------------------------------------------------------------------
# HF dataset
# ---------------------------------------------------------------------------

def make_hf_dataset(english, navajo):
    """Create HF dataset and log sample info."""
    print(f"\n[DEBUG] Creating HF dataset with {len(english)} examples")
    dataset = Dataset.from_dict({"source": english, "target": navajo})
    
    # Log a few random samples
    import random
    sample_indices = random.sample(range(len(english)), min(3, len(english)))
    print("[DEBUG] Sample entries from dataset:")
    for idx in sample_indices:
        print(f"  Sample {idx}:")
        print(f"    Source: {english[idx][:80]}...")
        print(f"    Target: {navajo[idx][:80]}...")
    
    return dataset


# ---------------------------------------------------------------------------
# 5-fold split - CORRECTED VERSION
# ---------------------------------------------------------------------------

def kfold_split(english, navajo, k=5, seed=42, output_dir=None):
    """
    Create k-fold splits where:
    - Each fold: train on k-1 folds, validate on half of remaining fold
    - The OTHER half of remaining fold is saved as test set (for later use)
    - Test portions across all folds combine to form complete test set
    - Each fold has DIFFERENT train/dev/test sets
    """
    n = len(english)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    
    # Shuffle the data based on indices
    english = [english[i] for i in indices]
    navajo = [navajo[i] for i in indices]
    
    # Split into k equal folds
    fold_size = n // k
    folds_eng = []
    folds_nav = []
    
    print(f"\n{'='*60}")
    print(f"[DEBUG] Creating {k}-fold splits from {n} total examples")
    print(f"[DEBUG] Random seed: {seed}")
    
    for fold_idx in range(k):
        start = fold_idx * fold_size
        end = start + fold_size if fold_idx < k - 1 else n
        folds_eng.append(english[start:end])
        folds_nav.append(navajo[start:end])
        print(f"[DEBUG] Fold {fold_idx} size: {len(folds_eng[fold_idx]):,} examples")
    
    # For each fold
    for fold_idx in range(k):
        # Training = all folds except current fold
        train_eng = []
        train_nav = []
        train_folds = []
        
        for f in range(k):
            if f != fold_idx:
                train_eng.extend(folds_eng[f])
                train_nav.extend(folds_nav[f])
                train_folds.append(f)
        
        # Current fold = split into dev (first half) and test (second half)
        current_eng = folds_eng[fold_idx]
        current_nav = folds_nav[fold_idx]
        
        mid = len(current_eng) // 2
        dev_eng = current_eng[:mid]
        dev_nav = current_nav[:mid]
        test_eng = current_eng[mid:]
        test_nav = current_nav[mid:]
        
        print(f"\n[DEBUG] Fold {fold_idx}:")
        print(f"  Train: {len(train_eng):,} examples (folds: {train_folds})")
        print(f"  Dev:   {len(dev_eng):,} examples (first half of fold {fold_idx})")
        print(f"  Test:  {len(test_eng):,} examples (SECOND HALF of fold {fold_idx} - saved for later)")
        print(f"  This fold's test set is DIFFERENT from other folds' test sets")
        
        # Save ALL split data for later use
        if output_dir:
            fold_dir = Path(output_dir) / f"fold_{fold_idx}"
            data_dir = fold_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the actual data files
            (data_dir / "train.eng").write_text("\n".join(train_eng), encoding="utf-8")
            (data_dir / "train.nav").write_text("\n".join(train_nav), encoding="utf-8")
            (data_dir / "dev.eng").write_text("\n".join(dev_eng), encoding="utf-8")
            (data_dir / "dev.nav").write_text("\n".join(dev_nav), encoding="utf-8")
            (data_dir / "test.eng").write_text("\n".join(test_eng), encoding="utf-8")
            (data_dir / "test.nav").write_text("\n".join(test_nav), encoding="utf-8")
            
            # Save metadata about the split
            split_dir = fold_dir / "splits"
            split_dir.mkdir(parents=True, exist_ok=True)
            
            with open(split_dir / "fold_assignment.json", "w") as f:
                json.dump({
                    "fold": fold_idx,
                    "train_folds": train_folds,
                    "dev_from_fold": fold_idx,
                    "dev_indices_in_fold": list(range(mid)),
                    "test_from_fold": fold_idx,
                    "test_indices_in_fold": list(range(mid, len(current_eng))),
                    "test_saved_for_later": True,
                    "note": "Test sets from all folds combine to form complete test set"
                }, f, indent=2)
            
            print(f"[DEBUG] Data saved to: {data_dir}")
            print(f"    - train.eng/train.nav ({len(train_eng)} pairs)")
            print(f"    - dev.eng/dev.nav ({len(dev_eng)} pairs)")
            print(f"    - test.eng/test.nav ({len(test_eng)} pairs) - FOR LATER TESTING")
        
        # Yield train, dev, and test (test is just for information, not used in training)
        yield (
            fold_idx,
            train_eng, train_nav,
            dev_eng, dev_nav,
            test_eng, test_nav,
        )
    
    # After all folds, show combined test set info
    print(f"\n{'='*60}")
    print("[DEBUG] TEST SET SUMMARY:")
    print("[DEBUG] The second half of each fold has been saved as test data.")
    print("[DEBUG] These test sets are ALL DIFFERENT from each other.")
    print("[DEBUG] To use the complete test set later, combine these files:")
    for fold_idx in range(k):
        print(f"    fold_{fold_idx}/data/test.eng and fold_{fold_idx}/data/test.nav")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fold(fold_idx, train_eng, train_nav, dev_eng, dev_nav, args, syllabifier=None):

    print(f"\n{'#'*80}")
    print(f"# FOLD {fold_idx} TRAINING STARTING")
    print(f"{'#'*80}")
    
    # GPU detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEBUG] Device: {device}")
    if torch.cuda.is_available():
        print(f"[DEBUG] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[DEBUG] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"[DEBUG] Current GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    fold_output = Path(args.output_dir) / f"fold_{fold_idx}"
    fold_output.mkdir(parents=True, exist_ok=True)
    print(f"\n[DEBUG] Fold output directory: {fold_output}")

    model_id = NLLB_MODELS[args.model_size]
    print(f"\n[DEBUG] Loading model: {model_id}")

    # Apply syllabification if requested
    if syllabifier:
        print(f"\n[DEBUG] Applying FOMA syllabification to training data...")
        print(f"[DEBUG] Sample before syllabification: {train_nav[0][:100]}...")
        train_nav = [syllabifier.syllabify_text(t) for t in train_nav]
        dev_nav = [syllabifier.syllabify_text(t) for t in dev_nav]
        print(f"[DEBUG] Sample after syllabification: {train_nav[0][:100]}...")
        print(f"[DEBUG] @ symbols in syllabified sample: {train_nav[0].count('@')}")
    else:
        print(f"\n[DEBUG] Syllabification disabled - using raw text")

    tokenizer = NllbTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    model.to(device)
    print(f"[DEBUG] Model moved to {device}")

    # Language setup
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = args.tgt_lang
    print(f"\n[DEBUG] Tokenizer configuration:")
    print(f"  Source language: {tokenizer.src_lang}")
    print(f"  Target language: {tokenizer.tgt_lang}")

    lang_id = tokenizer.convert_tokens_to_ids(args.tgt_lang)
    model.config.forced_bos_token_id = lang_id
    model.config.decoder_start_token_id = lang_id
    print(f"[DEBUG] Model decoder configuration:")
    print(f"  forced_bos_token_id: {model.config.forced_bos_token_id} ({args.tgt_lang})")
    print(f"  decoder_start_token_id: {model.config.decoder_start_token_id}")

    # Create datasets
    print(f"\n[DEBUG] Creating datasets...")
    train_ds = Dataset.from_dict({
        "source": train_eng,
        "target": train_nav
    })
    dev_ds = Dataset.from_dict({
        "source": dev_eng,
        "target": dev_nav
    })

    def preprocess(examples):
        # Encode source
        model_inputs = tokenizer(
            examples["source"],
            max_length=args.max_length,
            truncation=True,
        )

        # Encode target
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=args.max_length,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print(f"\n[DEBUG] Tokenizing datasets...")
    train_ds = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=["source", "target"],
        desc=f"Tokenizing fold {fold_idx} train"
    )

    dev_ds = dev_ds.map(
        preprocess,
        batched=True,
        remove_columns=["source", "target"],
        desc=f"Tokenizing fold {fold_idx} dev"
    )

    # Log tokenization stats
    print(f"\n[DEBUG] Tokenization stats:")
    print(f"  Train dataset size: {len(train_ds)}")
    print(f"  Dev dataset size: {len(dev_ds)}")
    
    # Check a few examples
    print(f"\n[DEBUG] Sample tokenized example:")
    for i in range(min(3, len(train_ds))):
        print(f"  Example {i}:")
        print(f"    input_ids length: {len(train_ds[i]['input_ids'])}")
        print(f"    labels length: {len(train_ds[i]['labels'])}")
        print(f"    attention_mask length: {len(train_ds[i]['attention_mask'])}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(fold_output),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,  # Avoid pickling issues
    )

    print(f"\n[DEBUG] Training arguments:")
    print(f"  Output directory: {training_args.output_dir}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Batch size (per device): {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  BF16 enabled: {training_args.bf16}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"\n[DEBUG] Starting training for fold {fold_idx}...")
    print(f"[DEBUG] Training samples: {len(train_ds)}")
    print(f"[DEBUG] Validation samples: {len(dev_ds)}")
    print(f"[DEBUG] NOTE: Test set from fold {fold_idx} is NOT being used")
    
    # Train with step-by-step logging
    train_result = trainer.train()
    
    print(f"\n[DEBUG] Training completed for fold {fold_idx}")
    print(f"[DEBUG] Best eval loss: {trainer.state.best_metric:.6f}")
    print(f"[DEBUG] Best model checkpoint: {trainer.state.best_model_checkpoint}")
    
    # Save training metrics
    metrics_path = fold_output / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "fold": fold_idx,
            "best_eval_loss": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "total_steps": trainer.state.global_step,
            "epoch": trainer.state.epoch,
        }, f, indent=2)
    print(f"[DEBUG] Training metrics saved to: {metrics_path}")

    return fold_output, trainer.state.best_metric


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_path", required=False, help="Path to Low_Resource_MT repo (required for syllabify)")
    parser.add_argument("--bible_eng_file", required=True)
    parser.add_argument("--bible_nav_file", required=True)
    parser.add_argument("--user_eng_file", required=True)
    parser.add_argument("--user_nav_file", required=True)
    parser.add_argument("--model_size", default="600M", choices=["600M", "3.3B"])
    parser.add_argument("--tgt_lang", default="nav_Latn")
    parser.add_argument("--output_dir", default="./eng_nav_nllb")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Re-run a single fold using pre-saved data splits
    parser.add_argument(
        "--fold_num",
        default="all",
        help=(
            "Which fold to run. Use 'all' to run all folds (default), "
            "or an integer (0-4) to re-run a single fold using the data "
            "splits already saved under <output_dir>/fold_<N>/data/."
        ),
    )

    # Syllabification toggle
    syllabify_group = parser.add_mutually_exclusive_group()
    syllabify_group.add_argument("--syllabify", action="store_true", help="Apply FOMA syllabification")
    syllabify_group.add_argument("--no_syllabify", action="store_true", help="Skip FOMA syllabification")
    parser.set_defaults(syllabify=False, no_syllabify=False)

    args = parser.parse_args()

    # Parse --fold_num
    if args.fold_num == "all":
        single_fold = None
    else:
        try:
            single_fold = int(args.fold_num)
        except ValueError:
            parser.error(f"--fold_num must be 'all' or an integer, got: '{args.fold_num}'")

    # Set default learning rate if not provided
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LR[args.model_size]
        print(f"\n[DEBUG] Using default learning rate for {args.model_size}: {args.learning_rate}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"\n{'='*80}")
    print("ENGLISH→NAVAJO TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"[DEBUG] Configuration:")
    print(f"  Model size: {args.model_size}")
    print(f"  Target language: {args.tgt_lang}")
    print(f"  Syllabification: {'ENABLED' if args.syllabify else 'DISABLED'}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Epochs per fold: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  K-folds: {args.k_folds}")
    print(f"  Random seed: {args.seed}")
    if single_fold is not None:
        print(f"  *** SINGLE FOLD MODE: re-running fold {single_fold} from saved splits ***")
    else:
        print(f"  Fold mode: ALL folds")
    print(f"{'='*80}")

    if not torch.cuda.is_available():
        print("\n[WARNING] No GPU detected — running on CPU (this will be very slow)")

    # Initialize syllabifier if requested
    syllabifier = None
    if args.syllabify:
        if not args.repo_path:
            raise ValueError("--repo_path is required when --syllabify is enabled")
        print("\n[DEBUG] Initializing FOMA syllabifier...")
        syllabifier = FOMASyllabifier(args.repo_path)

    fold_results = []

    # -----------------------------------------------------------------------
    # SINGLE FOLD MODE: load splits from disk, skip data loading & kfold_split
    # -----------------------------------------------------------------------
    if single_fold is not None:
        print(f"\n[DEBUG] Single-fold mode: loading fold {single_fold} splits from disk...")
        train_eng, train_nav, dev_eng, dev_nav, test_eng, test_nav = \
            load_fold_from_disk(single_fold, args.output_dir)

        print(f"\n{'*'*80}")
        print(f"* RE-RUNNING FOLD {single_fold} FROM SAVED SPLITS")
        print(f"{'*'*80}")

        best_path, best_loss = train_fold(
            single_fold,
            train_eng, train_nav,
            dev_eng, dev_nav,
            args,
            syllabifier,
        )
        fold_results.append((single_fold, best_loss, best_path))

        print(f"\n{'='*80}")
        print(f"FOLD {single_fold} RE-RUN COMPLETE")
        print(f"  eval_loss = {best_loss:.6f}")
        print(f"  model saved at: {best_path}")
        print(f"{'='*80}")

        # Update cv_results.json if it exists (patch this fold's entry)
        results_file = Path(args.output_dir) / "cv_results.json"
        if results_file.exists():
            with open(results_file) as f:
                cv = json.load(f)
            for entry in cv.get("all_folds", []):
                if entry["fold"] == single_fold:
                    entry["eval_loss"] = best_loss
                    entry["path"] = str(best_path)
                    break
            # Re-compute best fold
            best_entry = min(cv["all_folds"], key=lambda x: x["eval_loss"])
            cv["best_fold"] = best_entry["fold"]
            cv["best_eval_loss"] = best_entry["eval_loss"]
            cv["best_model_path"] = best_entry["path"]
            with open(results_file, "w") as f:
                json.dump(cv, f, indent=2)
            print(f"[DEBUG] cv_results.json updated with new fold {single_fold} result.")
        return

    # -----------------------------------------------------------------------
    # ALL FOLDS MODE: load raw data, run kfold_split, train each fold
    # -----------------------------------------------------------------------

    # Load data
    bible_eng, bible_nav = load_parallel_file(
        args.bible_eng_file, args.bible_nav_file, "Bible"
    )
    user_eng, user_nav = load_parallel_file(
        args.user_eng_file, args.user_nav_file, "User"
    )

    all_eng = bible_eng + user_eng
    all_nav = [normalize_text(t) for t in (bible_nav + user_nav)]

    print(f"\n[DEBUG] Combined dataset statistics:")
    print(f"  Bible pairs: {len(bible_eng):,}")
    print(f"  User pairs: {len(user_eng):,}")
    print(f"  Total pairs: {len(all_eng):,}")
    
    # Save full dataset for reference
    full_data_dir = Path(args.output_dir) / "full_dataset"
    full_data_dir.mkdir(parents=True, exist_ok=True)
    with open(full_data_dir / "all_eng.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_eng))
    with open(full_data_dir / "all_nav.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_nav))
    print(f"[DEBUG] Full dataset saved to: {full_data_dir}")

    # Run k-fold cross validation
    for fold_idx, train_eng, train_nav, dev_eng, dev_nav, test_eng, test_nav in \
            kfold_split(all_eng, all_nav, k=args.k_folds, seed=args.seed, output_dir=args.output_dir):

        print(f"\n{'*'*80}")
        print(f"* FOLD {fold_idx} STARTING")
        print(f"{'*'*80}")
        print(f"[DEBUG] Fold {fold_idx} data distribution:")
        print(f"  Training: {len(train_eng):,} examples (folds other than {fold_idx})")
        print(f"  Development: {len(dev_eng):,} examples (first half of fold {fold_idx})")
        print(f"  Test: {len(test_eng):,} examples (second half of fold {fold_idx} - SAVED, NOT USED)")

        # Train the fold (using ONLY train and dev)
        best_path, best_loss = train_fold(
            fold_idx,
            train_eng,
            train_nav,
            dev_eng,
            dev_nav,
            args,
            syllabifier,
        )
        
        fold_results.append((fold_idx, best_loss, best_path))
        
        # Show current ranking
        print(f"\n[DEBUG] Current fold results ranking:")
        sorted_results = sorted(fold_results, key=lambda x: x[1])
        for rank, (f_idx, loss, _) in enumerate(sorted_results):
            star = "*" if rank == 0 else " "
            print(f"  {star} Fold {f_idx}: eval_loss = {loss:.6f}")
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[DEBUG] GPU memory cleared after fold {fold_idx}")

    # Find best fold
    best_fold = min(fold_results, key=lambda x: x[1])

    print(f"\n{'='*80}")
    print("CROSS-VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"[DEBUG] All fold results:")
    for fold_idx, loss, path in fold_results:
        print(f"  Fold {fold_idx}: eval_loss = {loss:.6f}  |  model: {path}")
    
    print(f"\n[DEBUG] BEST FOLD: Fold {best_fold[0]} with eval_loss = {best_fold[1]:.6f}")
    print(f"[DEBUG] Best model path: {best_fold[2]}")
    
    print(f"\n[DEBUG] TEST SETS LOCATION:")
    print(f"  Each fold's test set is saved in: {args.output_dir}/fold_X/data/test.eng/.nav")
    print(f"  To create complete test set, combine all fold_X/data/test.eng files")
    print(f"  These test sets were NEVER used during training!")
    
    # Save overall results
    results_file = Path(args.output_dir) / "cv_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "best_fold": best_fold[0],
            "best_eval_loss": best_fold[1],
            "best_model_path": str(best_fold[2]),
            "all_folds": [
                {"fold": idx, "eval_loss": loss, "path": str(path)}
                for idx, loss, path in fold_results
            ],
            "test_sets": {
                "location": f"{args.output_dir}/fold_X/data/test.eng/.nav",
                "note": "Combine all fold test sets for complete test set",
                "never_used_in_training": True
            },
            "config": vars(args)
        }, f, indent=2)
    print(f"[DEBUG] Results saved to: {results_file}")


if __name__ == "__main__":
    main()
