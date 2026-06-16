#!/usr/bin/env python3
"""
train_nsl_m2m.py
================
NSL-MT for English → Navajo using M2M100.  Single-pool 5-fold CV.

LOSS — EXACTLY AS THE PAPER WRITES IT (Equations 2-4)
======================================================

Eq 3:  L_pos = -(1/|B_pos|) Σ log P(y|x;θ)        [mean CE over positives]
Eq 4:  L_neg = -(1/|B_neg|) Σ s(v)·log P(v|x;θ)   [severity-weighted mean CE over violations]
Eq 2:  L = L_pos + α·L_neg                          [ADD, not subtract]

Both L_pos and L_neg are standard cross-entropy values — positive scalars.
The combined loss is therefore always positive and bounded.

HOW THE ADDITION WORKS (the mechanism is subtle)
-------------------------------------------------
Adding α·L_neg does NOT directly "push" the model away from violations
via gradient direction in the conventional sense. Instead it works through
probabilistic competition:

  For the same source x, the batch contains:
    (x, y)   — correct translation → gradient says "increase P(y|x)"
    (x, v1)  — violation           → gradient says "increase P(v1|x)"
    (x, v2)  — violation           → gradient says "increase P(v2|x)"
    ...

  The softmax over all possible output sequences is a zero-sum game:
  P(y|x) + P(v1|x) + P(v2|x) + ... ≤ 1 (probability constraint).
  They compete for probability mass.

  Crucially, violations are DIVERSE and CHANGE every epoch (resampled).
  The model cannot simultaneously fit 4+ inconsistent corruptions of the
  same source. So the violation gradients cancel each other out over
  time, while the CONSISTENT signal from correct translations (same y
  every epoch) wins the competition.

  The severity weighting (s·CE for violations) amplifies this competition
  for the most grammatically damaging corruptions, giving the correct
  translation an even stronger relative advantage for those patterns.

WHY SUBTRACTION IS WRONG (what caused the collapse in the bad run)
------------------------------------------------------------------
L = L_pos - α·L_neg would try to DIRECTLY push P(violation) down by
maximizing CE(violation). But CE is unbounded from above. Early in training,
when the model has no Navajo knowledge, CE(violation) >> CE(correct), so:
  loss = 0.8 - 1.0 × (0.49 rebalance) × 5.0 = -1.65
Loss goes negative → gradient magnitudes explode → grad_norm of 831 →
training collapses. This is exactly what appeared in the logs.

IMPLEMENTATION DETAILS
-----------------------
  • α = 0.7 (paper default, robust in [0.3, 0.9] per Appendix B.1)
  • k ~ Uniform(3,5) violations per sentence per epoch (Algorithm 1)
  • Violations resampled every epoch (different seed) — prevents memorisation
  • Positives and violations shuffled together in each batch (Section 2.4)
  • Gradient clipping max_norm=1.0, AdamW, lr=2e-5, warmup=500 (Section 3.1)
  • eval_loss uses MLE only (dev batches get is_negative=None path)
  • No rebalance multiplier — the paper does not use one

DATA DESIGN
-----------
  Single unified pool: Bible + user + gloss sentences.
  5-fold CV with same seed as vanilla baseline → identical splits.
  Violations generated ONLY from train_records of each fold.
  Dev and test records NEVER enter the violation sampler.
  Test set contains only positive (eng, nav) pairs.

USAGE
-----
  python train_nsl_m2m.py \\
      --bible_eng_file  aligned_bibles/english_nlt.txt \\
      --bible_nav_file  aligned_bibles/navajo_nvjob.txt \\
      --user_eng_file   my_english.txt \\
      --user_nav_file   my_navajo.txt \\
      --nsl_gloss_file  xml_data.txt \\
      --output_dir      ./nsl_out

  # Single fold
  python train_nsl_m2m.py ... --fold_num 0
"""

import argparse
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    M2M100Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
)
import warnings
warnings.filterwarnings("ignore")

from navajo_violation_generators import (
    parse_gloss_file,
    make_untagged_record,
    _ALL_GENERATORS,
    SEVERITY,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
M2M_MODELS = {"418M": "facebook/m2m100_418M", "1.2B": "facebook/m2m100_1.2B"}
DEFAULT_LR  = {"418M": 2e-5, "1.2B": 1e-5}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u02bc", "'").strip()


def load_parallel(eng_path: str, nav_path: str, label: str):
    with open(eng_path, encoding="utf-8") as f:
        eng = [l.strip() for l in f if l.strip()]
    with open(nav_path, encoding="utf-8") as f:
        nav = [l.strip() for l in f if l.strip()]
    if len(eng) != len(nav):
        raise ValueError(f"{label}: {len(eng)} eng vs {len(nav)} nav")
    print(f"  {label}: {len(eng):,} pairs")
    return eng, nav


# ---------------------------------------------------------------------------
# K-fold splitting
# ---------------------------------------------------------------------------
def make_folds(items: list, k: int, seed: int) -> List[list]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    fold_size = len(shuffled) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end   = start + fold_size if i < k - 1 else len(shuffled)
        folds.append(shuffled[start:end])
    return folds


def get_train_dev_test(folds: list, fold_idx: int) -> Tuple[list, list, list]:
    """80% train / ~10% dev / ~10% test by splitting held-out fold in half."""
    train = []
    for i, fold in enumerate(folds):
        if i != fold_idx:
            train.extend(fold)
    held = folds[fold_idx]
    mid  = len(held) // 2
    return train, held[:mid], held[mid:]


# ---------------------------------------------------------------------------
# Violation sampling — Algorithm 1, lines 5-9
# ---------------------------------------------------------------------------
def sample_violations(records, k_min, k_max, epoch_seed):
    rng      = random.Random(epoch_seed)
    triplets = []

    for record in records:
        applicable = []
        for gen in _ALL_GENERATORS:
            result = gen(record)
            if result is not None:
                applicable.append(result)

        if not applicable:
            continue

        # Sample k ~ Uniform(k_min, k_max) — NOT capped by len(applicable)
        # If k > len(applicable), sample with replacement (Algorithm 1 line 7)
        k = rng.randint(k_min, k_max)

        if k <= len(applicable):
            chosen = rng.sample(applicable, k)
        else:
            # With replacement — same violation can appear multiple times
            # This matches Algorithm 1: t ~ Uniform(G) independently each draw
            chosen = rng.choices(applicable, k=k)

        for v in chosen:
            triplets.append({
                "source_en":   record["sentence_en"],
                "violated_nv": v["sentence"],
                "severity":    float(v["severity"]),
            })

    return triplets


# ---------------------------------------------------------------------------
# NSL Dataset — interleaved positives + negatives (Section 2.4)
# ---------------------------------------------------------------------------
class NSLDataset(TorchDataset):
    def __init__(self, pos_ds, neg_ds, seed: int = 42):
        self.pos = pos_ds
        self.neg = neg_ds
        indices  = [(False, i) for i in range(len(pos_ds))]
        indices += [(True,  i) for i in range(len(neg_ds))]
        random.Random(seed).shuffle(indices)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        is_neg, src = self.indices[idx]
        pool = self.neg if is_neg else self.pos
        item = {
            k: torch.tensor(v)
            for k, v in pool[src].items()
            if k in ("input_ids", "attention_mask", "labels")
        }
        item["is_negative"] = torch.tensor(is_neg)
        item["severity"]    = torch.tensor(
            float(self.neg[src]["severity"]) if is_neg else 0.0
        )
        return item


# ---------------------------------------------------------------------------
# Epoch callback — regenerates violations each epoch (Algorithm 1)
# ---------------------------------------------------------------------------
class NSLEpochCallback(TrainerCallback):
    """
    Rebuilds the training dataset with fresh violations at the start of each
    epoch. Seed = base_seed + epoch + 1, so epoch-0 init violations (seed =
    base_seed) are never reused.
    """
    def __init__(self, train_records, pos_ds, tokenizer,
                 max_length, base_seed=42, k_min=3, k_max=5):
        self.records   = train_records
        self.pos       = pos_ds
        self.tokenizer = tokenizer
        self.max_len   = max_length
        self.base_seed = base_seed
        self.k_min     = k_min
        self.k_max     = k_max
        self._trainer  = None  # injected after Trainer construction

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self._trainer is None:
            return
        epoch    = int(state.epoch) if state.epoch is not None else 0
        seed     = self.base_seed + epoch + 1
        triplets = sample_violations(self.records, self.k_min, self.k_max, seed)
        neg_ds   = tokenize_negatives(triplets, self.tokenizer, self.max_len)
        self._trainer.train_dataset = NSLDataset(self.pos, neg_ds, seed=seed)
        ratio = len(triplets) / max(1, len(self.pos))
        print(f"\n[NSL] Epoch {epoch + 1}: {len(triplets):,} violations "
              f"({ratio:.2f}:1 ratio, seed={seed})")


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------
class NSLDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        is_neg = [f.pop("is_negative") for f in features]
        sev    = [f.pop("severity")    for f in features]
        batch  = super().__call__(features)
        batch["is_negative"] = torch.stack([
            v if isinstance(v, torch.Tensor) else torch.tensor(v)
            for v in is_neg
        ])
        batch["severity"] = torch.stack([
            v if isinstance(v, torch.Tensor)
            else torch.tensor(v, dtype=torch.float32)
            for v in sev
        ])
        return batch


# ---------------------------------------------------------------------------
# NSL Trainer — paper-faithful loss: L = L_pos + α·L_neg
# ---------------------------------------------------------------------------
class NSLTrainer(Seq2SeqTrainer):
    """
    Implements Equations 2-4 from Keita et al. (2025) exactly as written.

    L_pos = mean CE over correct translations            (Eq 3)
    L_neg = severity-weighted mean CE over violations    (Eq 4)
    L     = L_pos + α · L_neg                           (Eq 2)

    The mechanism is probabilistic competition through the softmax constraint,
    NOT direct gradient penalization of violations (see module docstring).

    Loss is always positive → stable training.
    No rebalance multiplier — the paper does not use one.

    Dev batches (is_negative=None) use pure MLE for clean eval_loss.
    """
    def __init__(self, *args, nsl_alpha: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.nsl_alpha = nsl_alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        is_negative = inputs.pop("is_negative", None)
        severity    = inputs.pop("severity",    None)

        outputs = model(**inputs)
        logits  = outputs.logits   # (B, T, V)
        labels  = inputs["labels"] # (B, T)

        # Per-example mean CE (ignoring padding)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        b, t, v  = logits.shape
        per_tok  = loss_fct(logits.view(-1, v), labels.view(-1)).view(b, t)
        non_pad  = (labels != -100).float()
        per_ex   = (per_tok * non_pad).sum(1) / non_pad.sum(1).clamp(min=1)

        # Dev path — pure MLE for uncontaminated eval_loss
        if is_negative is None:
            loss = per_ex.mean()
            return (loss, outputs) if return_outputs else loss

        is_neg = is_negative.bool().to(per_ex.device)
        sev    = severity.float().to(per_ex.device)

        # L_pos — Eq 3: mean CE over correct translations
        if (~is_neg).any():
            l_pos = per_ex[~is_neg].mean()
        else:
            l_pos = torch.tensor(0.0, device=per_ex.device, requires_grad=True)

        # L_neg — Eq 4: severity-weighted mean CE over violations
        if is_neg.any():
            l_neg = (sev[is_neg] * per_ex[is_neg]).mean()
        else:
            l_neg = torch.tensor(0.0, device=per_ex.device)

        # L = L_pos + α·L_neg  — Eq 2, exactly as the paper writes it
        # Both terms are positive → loss is always positive → stable
        loss = l_pos + self.nsl_alpha * l_neg

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def tokenize_positives(hf_dataset, tokenizer, max_length: int, desc: str):
    def preprocess(examples):
        tokenizer.src_lang = "en"
        enc = tokenizer(examples["source"], max_length=max_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            lab = tokenizer(examples["target"], max_length=max_length, truncation=True)
        enc["labels"] = lab["input_ids"]
        return enc
    return hf_dataset.map(
        preprocess, batched=True,
        remove_columns=["source", "target"], desc=desc
    )


def tokenize_negatives(triplets: List[Dict], tokenizer, max_length: int) -> Dataset:
    if not triplets:
        return Dataset.from_dict(
            {"input_ids": [], "attention_mask": [], "labels": [], "severity": []}
        )
    tokenizer.src_lang = "en"
    enc = tokenizer(
        [t["source_en"]   for t in triplets],
        max_length=max_length, truncation=True, padding=False
    )
    with tokenizer.as_target_tokenizer():
        lab = tokenizer(
            [t["violated_nv"] for t in triplets],
            max_length=max_length, truncation=True, padding=False
        )
    return Dataset.from_dict({
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels":         lab["input_ids"],
        "severity":       [t["severity"] for t in triplets],
    })


# ---------------------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------------------
def train_fold(
    fold_idx: int,
    train_pairs: list,    # [(eng, nav), ...]
    dev_pairs:   list,
    test_pairs:  list,
    train_records: list,  # violation records — TRAIN SPLIT ONLY
    args,
) -> Tuple[Path, float]:

    print(f"\n{'#'*70}")
    print(f"# FOLD {fold_idx}  |  NSL-MT  |  α={args.nsl_alpha}  "
          f"k~U({args.nsl_k_min},{args.nsl_k_max})")
    print(f"  train={len(train_pairs):,}  dev={len(dev_pairs):,}  "
          f"test={len(test_pairs):,}  viol_records={len(train_records):,}")
    print(f"{'#'*70}")

    fold_dir = Path(args.output_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    data_dir = fold_dir / "data"
    data_dir.mkdir(exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id  = M2M_MODELS[args.model_size]
    tokenizer = M2M100Tokenizer.from_pretrained(model_id)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    lang_id   = tokenizer.get_lang_id(args.tgt_lang)
    model.config.forced_bos_token_id    = lang_id
    model.config.decoder_start_token_id = lang_id
    tokenizer.src_lang = "en"
    tokenizer.tgt_lang = args.tgt_lang
    model.to(device)

    train_eng, train_nav = zip(*train_pairs)
    dev_eng,   dev_nav   = zip(*dev_pairs)
    test_eng,  test_nav  = zip(*test_pairs)

    pos_ds = tokenize_positives(
        Dataset.from_dict({"source": list(train_eng), "target": list(train_nav)}),
        tokenizer, args.max_length, f"Fold {fold_idx}: tokenise train positives"
    )
    dev_ds = tokenize_positives(
        Dataset.from_dict({"source": list(dev_eng), "target": list(dev_nav)}),
        tokenizer, args.max_length, f"Fold {fold_idx}: tokenise dev"
    )
    # Dummy NSL fields on dev so collator can pop them;
    # is_negative=None path in compute_loss returns pure MLE loss
    dev_ds = dev_ds.map(
        lambda _: {"is_negative": False, "severity": 0.0},
        desc="Dummy NSL fields on dev"
    )

    # Epoch-0 violations (callback replaces these at start of epoch 1)
    init_trips = sample_violations(
        train_records, args.nsl_k_min, args.nsl_k_max, args.seed
    )
    init_neg = tokenize_negatives(init_trips, tokenizer, args.max_length)
    ratio    = len(init_trips) / max(1, len(pos_ds))
    print(f"  Epoch-0 violations: {len(init_trips):,}  ({ratio:.2f}:1 ratio)")

    train_ds      = NSLDataset(pos_ds, init_neg, seed=args.seed)
    data_collator = NSLDataCollator(
        tokenizer=tokenizer, model=model,
        padding=True, label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(fold_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        warmup_steps=500,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        bf16=torch.cuda.is_available(),
        max_grad_norm=1.0,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,   # MUST be False — keeps is_negative/severity
        predict_with_generate=False,
    )

    epoch_cb = NSLEpochCallback(
        train_records=train_records,
        pos_ds=pos_ds,
        tokenizer=tokenizer,
        max_length=args.max_length,
        base_seed=args.seed,
        k_min=args.nsl_k_min,
        k_max=args.nsl_k_max,
    )

    trainer = NSLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        nsl_alpha=args.nsl_alpha,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            epoch_cb,
        ],
    )
    epoch_cb._trainer = trainer

    trainer.train()

    best_loss = trainer.state.best_metric
    best_ckpt = trainer.state.best_model_checkpoint
    print(f"\n  Fold {fold_idx} done — best eval_loss={best_loss:.4f}  "
          f"ckpt={best_ckpt}")

    # Save splits — test.eng/test.nav are positive pairs only
    def wl(path, lines):
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    wl(data_dir / "train.eng", list(train_eng))
    wl(data_dir / "train.nav", list(train_nav))
    wl(data_dir / "dev.eng",   list(dev_eng))
    wl(data_dir / "dev.nav",   list(dev_nav))
    wl(data_dir / "test.eng",  list(test_eng))
    wl(data_dir / "test.nav",  list(test_nav))

    with open(data_dir / "violations_epoch0.jsonl", "w", encoding="utf-8") as f:
        for t in init_trips:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    with open(fold_dir / "metrics.json", "w") as f:
        json.dump({
            "fold":            fold_idx,
            "best_eval_loss":  best_loss,
            "best_checkpoint": best_ckpt,
            "total_steps":     trainer.state.global_step,
            "nsl_alpha":       args.nsl_alpha,
            "nsl_k_min":       args.nsl_k_min,
            "nsl_k_max":       args.nsl_k_max,
            "train_pairs":     len(train_pairs),
            "dev_pairs":       len(dev_pairs),
            "test_pairs":      len(test_pairs),
            "viol_records":    len(train_records),
            "epoch0_viols":    len(init_trips),
            "epoch0_ratio":    round(ratio, 3),
        }, f, indent=2)

    return fold_dir, best_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NSL-MT English→Navajo, single-pool 5-fold CV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bible_eng_file", required=True)
    parser.add_argument("--bible_nav_file", required=True)
    parser.add_argument("--user_eng_file",  required=True)
    parser.add_argument("--user_nav_file",  required=True)
    parser.add_argument("--nsl_gloss_file", required=True)
    parser.add_argument("--nsl_alpha",  type=float, default=0.7,
                        help="α weight for violation loss. Paper: 0.7, robust [0.3,0.9].")
    parser.add_argument("--nsl_k_min",  type=int,   default=3)
    parser.add_argument("--nsl_k_max",  type=int,   default=5,
                        help="Use 5-7 for 6:1 ratio (Appendix B.2: +13.1%% BLEU).")
    parser.add_argument("--model_size", default="418M", choices=["418M", "1.2B"])
    parser.add_argument("--tgt_lang",   default="nav")
    parser.add_argument("--output_dir", default="./nsl_out")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--grad_accum", type=int,   default=4)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_length", type=int,   default=256)
    parser.add_argument("--k_folds",    type=int,   default=5)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--fold_num",   default="all",
                        help="'all' or integer fold index.")
    args = parser.parse_args()

    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LR[args.model_size]

    single_fold = None if args.fold_num == "all" else int(args.fold_num)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*70}")
    print("NSL-MT  |  English → Navajo  |  Single-pool 5-fold CV")
    print(f"{'='*70}")
    print(f"  Model  : M2M100-{args.model_size}")
    print(f"  α      : {args.nsl_alpha}  (paper: 0.7)")
    print(f"  k      : Uniform({args.nsl_k_min},{args.nsl_k_max})")
    print(f"  Loss   : L = L_pos + α·L_neg  (paper Eq 2, stable addition)")
    print(f"  Output : {args.output_dir}")
    print(f"{'='*70}")

    # Load data
    print("\n[1] Loading parallel data...")
    bible_eng, bible_nav = load_parallel(args.bible_eng_file, args.bible_nav_file, "Bible")
    user_eng,  user_nav  = load_parallel(args.user_eng_file,  args.user_nav_file,  "User")

    print("\n[2] Parsing gloss file...")
    gloss_records = parse_gloss_file(args.nsl_gloss_file)
    print(f"  Gloss sentences: {len(gloss_records):,}")
    gloss_eng = [r["sentence_en"] for r in gloss_records]
    gloss_nav = [normalize(r["sentence_nv"]) for r in gloss_records]
    for r, n in zip(gloss_records, gloss_nav):
        r["sentence_nv"] = n

    # Build unified pool of (eng, nav, record) triples
    print("\n[3] Building unified pool...")
    bible_recs = [make_untagged_record(e, normalize(n))
                  for e, n in zip(bible_eng, bible_nav)]
    user_recs  = [make_untagged_record(e, normalize(n))
                  for e, n in zip(user_eng,  user_nav)]

    all_triples = (
        [(e, normalize(n), r) for e, n, r in zip(bible_eng, bible_nav, bible_recs)] +
        [(e, normalize(n), r) for e, n, r in zip(user_eng,  user_nav,  user_recs)]  +
        [(e, n,            r) for e, n, r in zip(gloss_eng, gloss_nav, gloss_records)]
    )
    print(f"  Bible: {len(bible_eng):,}  User: {len(user_eng):,}  "
          f"Gloss: {len(gloss_records):,}  Total: {len(all_triples):,}")

    # K-fold
    print(f"\n[4] {args.k_folds}-fold split (seed={args.seed})...")
    folds = make_folds(all_triples, args.k_folds, args.seed)
    print(f"  Fold sizes: {[len(f) for f in folds]}")

    # Train
    fold_results = []
    to_run = [single_fold] if single_fold is not None else list(range(args.k_folds))

    for fold_idx in to_run:
        train_t, dev_t, test_t = get_train_dev_test(folds, fold_idx)

        train_pairs   = [(e, n) for e, n, _ in train_t]
        dev_pairs     = [(e, n) for e, n, _ in dev_t]
        test_pairs    = [(e, n) for e, n, _ in test_t]
        train_records = [r       for _, _, r in train_t]   # TRAIN ONLY

        fold_dir = Path(args.output_dir) / f"fold_{fold_idx}"
        data_dir = fold_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save full clean test set (NO violations)
        with open(data_dir / "test_readable.txt", "w", encoding="utf-8") as f:
            for e, n in test_pairs:
                f.write(f"EN: {e}\nNV: {n}\n\n")

        # Optional: also save standard format
        (data_dir / "test.eng").write_text(
            "\n".join(e for e, _ in test_pairs), encoding="utf-8"
        )
        (data_dir / "test.nav").write_text(
            "\n".join(n for _, n in test_pairs), encoding="utf-8"
        )

        best_dir, best_loss = train_fold(
            fold_idx, train_pairs, dev_pairs, test_pairs,
            train_records, args
        )
        fold_results.append((fold_idx, best_loss, best_dir))

        print("\n  Results so far:")
        for fi, fl, _ in fold_results:
            best_m = " ← best" if fl == min(r[1] for r in fold_results) else ""
            print(f"    Fold {fi}: eval_loss={fl:.4f}{best_m}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    if len(fold_results) == args.k_folds:
        best = min(fold_results, key=lambda x: x[1])
        print(f"\n{'='*70}")
        print("CROSS-VALIDATION COMPLETE")
        for fi, fl, _ in fold_results:
            print(f"  Fold {fi}: eval_loss={fl:.4f}"
                  + (" ← BEST" if fi == best[0] else ""))
        out = Path(args.output_dir) / "cv_results.json"
        out.write_text(json.dumps({
            "best_fold":      best[0],
            "best_eval_loss": best[1],
            "best_path":      str(best[2]),
            "nsl_alpha":      args.nsl_alpha,
            "nsl_k_min":      args.nsl_k_min,
            "nsl_k_max":      args.nsl_k_max,
            "all_folds": [{"fold": fi, "eval_loss": fl, "path": str(fp)}
                          for fi, fl, fp in fold_results],
            "config": vars(args),
        }, indent=2))
        print(f"\n  Saved → {out}\n{'='*70}")


if __name__ == "__main__":
    main()
