#!/usr/bin/env python3
"""
test_nsl_m2m.py — Translate test set, write predictions for score.py.

Usage:
  python test_nsl_m2m.py \\
      --checkpoint_dir ./nsl_out/fold_0/checkpoint-XXXX \\
      --data_dir       ./nsl_out/fold_0/data \\
      --output_dir     ./nsl_out/fold_0/results
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, M2M100Tokenizer


def translate_batch(texts, model, tokenizer, device, src_lang, tgt_lang, max_length):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(texts, return_tensors="pt",
                       padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            max_length=max_length, num_beams=5, early_stopping=True,
        )
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_dir",       required=True)
    parser.add_argument("--output_dir",     default=None)
    parser.add_argument("--src_lang",       default="en")
    parser.add_argument("--tgt_lang",       default="nav")
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--max_length",     type=int, default=256)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    eng  = [l.strip() for l in (data_dir / "test.eng").read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = [l.strip() for l in (data_dir / "test.nav").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(eng) == len(gold)

    print(f"  Test : {len(eng):,} sentences")
    tokenizer = M2M100Tokenizer.from_pretrained(args.checkpoint_dir)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_dir)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    preds = []
    for i in range(0, len(eng), args.batch_size):
        preds.extend(translate_batch(
            eng[i:i + args.batch_size], model, tokenizer,
            device, args.src_lang, args.tgt_lang, args.max_length
        ))
        done = min(i + args.batch_size, len(eng))
        if done % (args.batch_size * 10) == 0 or done == len(eng):
            print(f"  Translated {done}/{len(eng)}")

    out = output_dir / "test_predictions.txt"
    with open(out, "w", encoding="utf-8") as f:
        for e, g, p in zip(eng, gold, preds):
            f.write(f"English   : {e}\nGold (raw): {g}\nGold (det): {g}\n"
                    f"Pred (raw): {p}\nPred (det): {p}\n\n")
    print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
