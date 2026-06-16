#!/usr/bin/env python3
"""
Probe ALL M2M100 language tags for zero-shot translation.
Finds the closest language for Navajo using chrF / BLEU on a sample.
Saves results to CSV and prints the top 3 tags by chrF.

Usage:
  python probe_m2m_langs.py \
      --eng_file sample.eng \
      --nav_file sample.nav \
      --n_sentences 50 \
      --output_csv results.csv
"""

from transformers import AutoModelForSeq2SeqLM, M2M100Tokenizer
import torch
import sacrebleu
import argparse
import csv
from pathlib import Path


def translate_batch(model, tokenizer, sentences, tgt_lang, max_length=256, batch_size=8):
    try:
        tgt_lang_id = tokenizer.get_lang_id(tgt_lang)
    except KeyError:
        return None  # tag not in vocab, skip

    model.config.forced_bos_token_id = tgt_lang_id
    all_outputs = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        tokenizer.src_lang = "en"
        tokenizer.tgt_lang = tgt_lang
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=max_length,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        all_outputs.extend(decoded)

    return all_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eng_file",    required=True)
    parser.add_argument("--nav_file",    required=True)
    parser.add_argument("--n_sentences", type=int, default=50,
                        help="How many sentence pairs to probe (default: 50)")
    parser.add_argument("--model_size",  default="418M", choices=["418M", "1.2B"])
    parser.add_argument("--output_csv",  default=None,
                        help="Path to save results as CSV (default: probe_results.csv)")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = "probe_results.csv"

    model_id = {
        "418M": "facebook/m2m100_418M",
        "1.2B": "facebook/m2m100_1.2B",
    }[args.model_size]

    print(f"Loading {model_id}...")
    tokenizer = M2M100Tokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")

    with open(args.eng_file, encoding="utf-8") as f:
        eng = [l.strip() for l in f if l.strip()][:args.n_sentences]
    with open(args.nav_file, encoding="utf-8") as f:
        nav = [l.strip() for l in f if l.strip()][:args.n_sentences]

    # Get ALL language codes from the tokenizer vocab, excluding English
    candidate_langs = [
        lang for lang in tokenizer.lang_code_to_id.keys()
        if lang != "en"
    ]

    print(f"\nProbing {len(eng)} sentence pairs across {len(candidate_langs)} language tags\n")
    print(f"{'Lang tag':<14} {'chrF':>10}  {'BLEU':>10}  {'Status'}")
    print("-" * 55)

    results = []
    csv_rows = []

    for lang in sorted(candidate_langs):
        hypotheses = translate_batch(model, tokenizer, eng, lang)
        if hypotheses is None:
            print(f"{lang:<14} {'N/A':>10}  {'N/A':>10}  ← tag not in M2M100 vocab")
            csv_rows.append({
                "lang_tag": lang,
                "chrf":     "N/A",
                "bleu":     "N/A",
                "status":   "not in vocab",
            })
            continue

        chrf = sacrebleu.corpus_chrf(hypotheses, [nav]).score
        bleu = sacrebleu.corpus_bleu(hypotheses, [nav]).score
        results.append((lang, chrf, bleu, hypotheses))
        csv_rows.append({
            "lang_tag": lang,
            "chrf":     f"{chrf:.5f}",
            "bleu":     f"{bleu:.5f}",
            "status":   "ok",
        })
        print(f"{lang:<14} {chrf:>10.5f}  {bleu:>10.5f}")

    # Write CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lang_tag", "chrf", "bleu", "status"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nResults saved to: {csv_path}")

    if results:
        top3 = sorted(results, key=lambda x: x[1], reverse=True)[:3]
        print("\n→ Top 3 tags by chrF:")
        for i, (lang, chrf, bleu, hyp) in enumerate(top3, 1):
            print(f"  {i}. {lang}  (chrF={chrf:.5f}, BLEU={bleu:.5f})")
            print(f"     Sample: {hyp[0] if hyp else 'N/A'}")


if __name__ == "__main__":
    main()
