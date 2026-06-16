#!/usr/bin/env python3
"""
Translate English → Target language using a trained NLLB checkpoint.
No scoring. Saves formatted predictions to file.
"""

import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path


# ---------------------------------------------------------------------------
# Detokenization
# ---------------------------------------------------------------------------

def detokenize(text: str) -> str:
    text = text.replace("@ @", "")
    tokens = text.split()
    tokens = [t.strip("@") for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(data_dir):
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

def translate_batch(texts, model, tokenizer, device, lang_trg, max_length=128):
    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    tgt_lang_token = tokenizer.convert_tokens_to_ids(lang_trg)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_token,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )

    translations = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )
    return translations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def translate_model(checkpoint_dir, data_dir, lang_trg, batch_size, output_file):

    print("=" * 80)
    print(f"Checkpoint  : {checkpoint_dir}")
    print(f"Data dir    : {data_dir}")
    print(f"Target lang : {lang_trg}")
    print(f"Output file : {output_file}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Load data
    print("\nLoading test data...")
    english_test, gold_test = load_test_data(data_dir)
    print(f"Loaded {len(english_test)} sentence pairs")

    # Translate
    print("\nTranslating...")
    all_preds_raw = []

    for i in range(0, len(english_test), batch_size):
        batch = english_test[i:i + batch_size]
        preds = translate_batch(batch, model, tokenizer, device, lang_trg)
        all_preds_raw.extend(preds)

        print(f"  Translated {min(i + batch_size, len(english_test))}/{len(english_test)}")

    all_preds_clean = [detokenize(p) for p in all_preds_raw]
    gold_clean = [detokenize(g) for g in gold_test]

    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ref, gold_raw, gold_c, pred_raw, pred_c in zip(
            english_test, gold_test, gold_clean, all_preds_raw, all_preds_clean
        ):
            f.write(
                f"English   : {ref}\n"
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
        description="Translate English → target language (no evaluation)"
    )
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to saved model checkpoint")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing test.eng and test.nav")
    parser.add_argument("--lang_trg", default="fin_Latn",
                        help="NLLB target language code")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_file", default="test_preds.txt",
                        help="Where to save formatted predictions")

    args = parser.parse_args()

    translate_model(
        args.checkpoint_dir,
        args.data_dir,
        args.lang_trg,
        args.batch_size,
        args.output_file,
    )


if __name__ == "__main__":
    main()
