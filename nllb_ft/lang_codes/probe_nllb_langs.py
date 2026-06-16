#!/usr/bin/env python3
"""
Probe all NLLB-200 language tags for zero-shot translation.
Finds the closest language for Navajo using chrF / BLEU on a sample.
Saves results to CSV and prints the top 3 tags by chrF.
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import sacrebleu
import argparse
import csv

def translate_batch(model, tokenizer, sentences, tgt_lang, max_length=256, batch_size=8):
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if tgt_lang_id == tokenizer.unk_token_id:
        return None  # tag not in vocab, skip
    model.config.forced_bos_token_id = tgt_lang_id
    all_outputs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        tokenizer.src_lang = "eng_Latn"
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(model.device)
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
    parser.add_argument('--eng_file',     required=True)
    parser.add_argument('--nav_file',     required=True)
    parser.add_argument('--n_sentences',  type=int, default=50)
    parser.add_argument('--model_size',   default='600M', choices=['600M', '3.3B'])
    parser.add_argument('--output_csv',   default='probe_results.csv')
    args = parser.parse_args()

    model_id = {
        "600M": "facebook/nllb-200-distilled-600M",
        "3.3B": "facebook/nllb-200-3.3B",
    }[args.model_size]

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")

    # Load sample sentences
    with open(args.eng_file, encoding='utf-8') as f:
        eng = [l.strip() for l in f if l.strip()][:args.n_sentences]
    with open(args.nav_file, encoding='utf-8') as f:
        nav = [l.strip() for l in f if l.strip()][:args.n_sentences]

    print(f"\nProbing {len(eng)} sentence pairs across all NLLB-200 language tags\n")
    print(f"{'Lang tag':<14} {'chrF':>10}  {'BLEU':>10}  {'Status'}")
    print("-" * 55)

    # All NLLB-200 language tags except English
    #candidate_langs = [tok for tok in tokenizer.lang_code_to_id.keys() if tok != "eng_Latn"]
    candidate_langs = [tok for tok in tokenizer.additional_special_tokens if tok != "eng_Latn"]

    results = []
    for lang in candidate_langs:
        hypotheses = translate_batch(model, tokenizer, eng, lang)
        if hypotheses is None:
            print(f"{lang:<14} {'N/A':>10}  {'N/A':>10}  ← not in vocab")
            continue
        chrf = sacrebleu.corpus_chrf(hypotheses, [nav]).score
        bleu = sacrebleu.corpus_bleu(hypotheses, [nav]).score
        results.append((lang, chrf, bleu, hypotheses))
        print(f"{lang:<14} {chrf:>10.5f}  {bleu:>10.5f}")

    # Save CSV
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["lang", "chrF", "BLEU"])
        for lang, chrf, bleu, _ in results:
            writer.writerow([lang, f"{chrf:.5f}", f"{bleu:.5f}"])

    print(f"\nResults saved to: {args.output_csv}")

    if results:
        top5 = sorted(results, key=lambda x: x[1], reverse=True)[:5]  # top 5 by chrF
        print("\n→ Top 5 tags by chrF:")
        for i, (lang, chrf, bleu, hyp) in enumerate(top3, 1):
            print(f"  {i}. {lang}  (chrF={chrf:.5f}, BLEU={bleu:.5f})")
            print(f"    Sample output: {hyp[0] if hyp else 'N/A'}")

if __name__ == '__main__':
    main()
