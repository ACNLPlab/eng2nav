#!/usr/bin/env python3
"""
test_nsl_nllb.py — Translate test set with a fine-tuned NLLB-200 checkpoint,
                   write predictions for score.py.

NLLB vs M2M differences
------------------------
  • Tokenizer : NllbTokenizer (not M2M100Tokenizer)
  • Lang codes: BCP-47 style — "eng_Latn", "nav_Latn"  (not "en", "nav")
  • forced_bos: tokenizer.lang_code_to_id[tgt_lang]    (not get_lang_id())

Usage:
  python test_nsl_nllb.py \\
      --checkpoint_dir ./nsl_nllb_out/fold_0/checkpoint-XXXX \\
      --data_dir       ./nsl_nllb_out/fold_0/data \\
      --output_dir     ./nsl_nllb_out/fold_0/results
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer


def get_forced_bos_token_id(tokenizer, tgt_lang: str) -> int:
    """
    NLLB stores language token IDs in lang_code_to_id.
    Raises a clear error if the language code is not found.
    """
    if hasattr(tokenizer, "lang_code_to_id"):
        if tgt_lang in tokenizer.lang_code_to_id:
            return tokenizer.lang_code_to_id[tgt_lang]
        raise ValueError(
            f"Language code '{tgt_lang}' not found in tokenizer.lang_code_to_id.\n"
            f"Use a BCP-47 NLLB code such as 'nav_Latn' or 'eng_Latn'.\n"
            f"Available codes (sample): "
            + str(list(tokenizer.lang_code_to_id.keys())[:10])
        )
    # Fallback: treat the lang string as a special token
    return tokenizer.convert_tokens_to_ids(tgt_lang)


def translate_batch(texts, model, tokenizer, device,
                    src_lang: str, tgt_lang: str,
                    max_length: int, forced_bos_token_id: int):
    """
    Tokenize a batch with src_lang, generate with forced_bos_token_id for tgt_lang.
    """
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Translate test set with fine-tuned NLLB-200 and write predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to a saved NLLB checkpoint directory.")
    parser.add_argument("--data_dir",       required=True,
                        help="Directory containing test.eng and test.nav.")
    parser.add_argument("--output_dir",     default=None,
                        help="Where to write test_predictions.txt "
                             "(default: <data_dir>/../results).")
    parser.add_argument("--src_lang",       default="eng_Latn",
                        help="NLLB BCP-47 source language code (default: eng_Latn).")
    parser.add_argument("--tgt_lang",       default="nav_Latn",
                        help="NLLB BCP-47 target language code (default: nav_Latn).")
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--max_length",     type=int, default=256)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else data_dir.parent / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    eng  = [
        l.strip()
        for l in (data_dir / "test.eng").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    gold = [
        l.strip()
        for l in (data_dir / "test.nav").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(eng) == len(gold), (
        f"Mismatch: {len(eng)} English lines vs {len(gold)} Navajo lines"
    )

    print(f"  Test sentences : {len(eng):,}")
    print(f"  Checkpoint     : {args.checkpoint_dir}")
    print(f"  {args.src_lang} → {args.tgt_lang}")

    # Load tokenizer and model from the fine-tuned checkpoint
    tokenizer = NllbTokenizer.from_pretrained(
        args.checkpoint_dir, src_lang=args.src_lang
    )
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_dir)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Resolve forced_bos once — reused for every batch
    forced_bos = get_forced_bos_token_id(tokenizer, args.tgt_lang)
    print(f"  forced_bos_token_id = {forced_bos}  ({args.tgt_lang})")

    preds = []
    for i in range(0, len(eng), args.batch_size):
        batch = eng[i : i + args.batch_size]
        preds.extend(
            translate_batch(
                batch, model, tokenizer, device,
                args.src_lang, args.tgt_lang,
                args.max_length, forced_bos,
            )
        )
        done = min(i + args.batch_size, len(eng))
        if done % (args.batch_size * 10) == 0 or done == len(eng):
            print(f"  Translated {done}/{len(eng)}")

    out = output_dir / "test_predictions.txt"
    with open(out, "w", encoding="utf-8") as f:
        for e, g, p in zip(eng, gold, preds):
            f.write(
                f"English   : {e}\n"
                f"Gold (raw): {g}\n"
                f"Gold (det): {g}\n"
                f"Pred (raw): {p}\n"
                f"Pred (det): {p}\n\n"
            )
    print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
