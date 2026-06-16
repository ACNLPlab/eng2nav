#!/usr/bin/env python3
"""
Translate English → Target language using a trained M2M100 checkpoint.
No scoring. Saves formatted predictions to file.
"""

import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, M2M100Tokenizer
from pathlib import Path


# ---------------------------------------------------------------------------
# Detokenization
# ---------------------------------------------------------------------------

def detokenize(text: str) -> str:
    """Remove FOMA syllable markers."""
    text = text.replace("@ @", "")
    tokens = text.split()
    tokens = [t.strip("@") for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(data_dir):
    """Load test.eng and test.nav files."""
    test_eng_file = Path(data_dir) / "test.eng"
    test_nav_file = Path(data_dir) / "test.nav"

    with open(test_eng_file, "r", encoding="utf-8") as f:
        english = [line.strip() for line in f]

    with open(test_nav_file, "r", encoding="utf-8") as f:
        gold = [line.strip() for line in f]

    return english, gold


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_batch(texts, model, tokenizer, device, src_lang, tgt_lang, max_length=128):
    """
    Translate a batch of texts using M2M100.
    
    M2M100 differences from NLLB:
    - Uses M2M100Tokenizer (not AutoTokenizer)
    - Language codes: "en", "fr", "sw" (not "eng_Latn", "fra_Latn")
    - Uses tokenizer.get_lang_id() (not convert_tokens_to_ids)
    """
    
    # Set source language
    tokenizer.src_lang = src_lang

    # Tokenize input
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get target language ID
    tgt_lang_id = tokenizer.get_lang_id(tgt_lang)

    # Generate translations
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )

    # Decode
    translations = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )
    return translations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def translate_model(checkpoint_dir, data_dir, src_lang, tgt_lang, batch_size, output_file):

    print("=" * 80)
    print("M2M100 TRANSLATION")
    print("=" * 80)
    print(f"Checkpoint  : {checkpoint_dir}")
    print(f"Data dir    : {data_dir}")
    print(f"Source lang : {src_lang}")
    print(f"Target lang : {tgt_lang}")
    print(f"Output file : {output_file}")
    print("=" * 80)

    # Load model - use M2M100Tokenizer instead of AutoTokenizer
    print("\nLoading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
    tokenizer = M2M100Tokenizer.from_pretrained(checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Load test data
    print("\nLoading test data...")
    english_test, gold_test = load_test_data(data_dir)
    print(f"Loaded {len(english_test)} sentence pairs")

    # Translate
    print("\nTranslating...")
    all_preds_raw = []

    for i in range(0, len(english_test), batch_size):
        batch = english_test[i:i + batch_size]
        preds = translate_batch(batch, model, tokenizer, device, src_lang, tgt_lang)
        all_preds_raw.extend(preds)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Translated {min(i + batch_size, len(english_test))}/{len(english_test)}")

    # Detokenize (remove @@ markers if syllabified)
    all_preds_clean = [detokenize(p) for p in all_preds_raw]
    gold_clean = [detokenize(g) for g in gold_test]

    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for eng, gold_raw, gold_c, pred_raw, pred_c in zip(
            english_test, gold_test, gold_clean, all_preds_raw, all_preds_clean
        ):
            f.write(
                f"English   : {eng}\n"
                f"Gold (raw): {gold_raw}\n"
                f"Gold (det): {gold_c}\n"
                f"Pred (raw): {pred_raw}\n"
                f"Pred (det): {pred_c}\n\n"
            )

    print("\nDone.")
    print(f"Saved predictions to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Translate English → target language using M2M100 (no evaluation)"
    )
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to saved M2M100 model checkpoint")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing test.eng and test.nav")
    parser.add_argument("--src_lang", default="en",
                        help="M2M100 source language code (e.g., 'en')")
    parser.add_argument("--tgt_lang", default="sw",
                        help="M2M100 target language code (e.g., 'sw', 'nav', 'grn')")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_file", default="test_preds_m2m100.txt",
                        help="Where to save formatted predictions")

    args = parser.parse_args()

    translate_model(
        args.checkpoint_dir,
        args.data_dir,
        args.src_lang,
        args.tgt_lang,
        args.batch_size,
        args.output_file,
    )


if __name__ == "__main__":
    main()
