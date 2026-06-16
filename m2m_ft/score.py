#!/usr/bin/env python3
"""
Evaluate English→Navajo predictions
Computes per-sentence BLEU, chrF, chrF++ scores for both raw and det variants.
"""
import argparse
import re
from pathlib import Path
from sacrebleu.metrics import BLEU, CHRF


def untokenize(text: str) -> str:
    text = text.replace("@@ ", "")
    text = text.replace("@@", "")
    text = text.replace("@ ", "")
    text = text.replace(" @", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_predictions_file(filepath: str):
    """
    Parse predictions file with format:
        English   : ...
        Pred (raw): ...
        Pred (det): ...
        Gold (raw): ...
        Gold (det): ...
    Returns list of dicts with keys: english, pred_raw, pred_det, gold_raw, gold_det
    """
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        english = pred_raw = pred_det = gold_raw = gold_det = None
        for line in block.split('\n'):
            if line.startswith('English'):
                english = line.split(':', 1)[1].strip()
            elif line.startswith('Pred (raw)'):
                pred_raw = line.split(':', 1)[1].strip()
            elif line.startswith('Pred (det)'):
                pred_det = line.split(':', 1)[1].strip()
            elif line.startswith('Gold (raw)'):
                gold_raw = line.split(':', 1)[1].strip()
            elif line.startswith('Gold (det)'):
                gold_det = line.split(':', 1)[1].strip()

        if all(v is not None for v in [english, pred_raw, pred_det, gold_raw, gold_det]):
            entries.append({
                'english':  english,
                'pred_raw': pred_raw,
                'pred_det': pred_det,
                'gold_raw': gold_raw,
                'gold_det': gold_det,
            })

    return entries


def score_sentence(pred: str, gold: str):
    bleu   = BLEU(effective_order=True)
    chrf   = CHRF(word_order=0)
    chrfpp = CHRF(word_order=2)
    return (
        bleu.sentence_score(pred, [gold]).score,
        chrf.sentence_score(pred, [gold]).score,
        chrfpp.sentence_score(pred, [gold]).score,
    )


def evaluate(input_file: str, output_file: str):
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")

    entries = parse_predictions_file(input_file)
    print(f"Found {len(entries)} entries")

    if not entries:
        print("ERROR: No entries found!")
        return

    results = []
    for entry in entries:
        pred_raw = untokenize(entry['pred_raw'])
        pred_det = untokenize(entry['pred_det'])
        gold_raw = untokenize(entry['gold_raw'])
        gold_det = untokenize(entry['gold_det'])

        bleu_raw, chrf_raw, chrfpp_raw = score_sentence(pred_raw, gold_raw)
        bleu_det, chrf_det, chrfpp_det = score_sentence(pred_det, gold_det)

        results.append({
            'english':   entry['english'],
            'pred_raw':  pred_raw,
            'pred_det':  pred_det,
            'gold_raw':  gold_raw,
            'gold_det':  gold_det,
            'bleu_raw':  bleu_raw,
            'chrf_raw':  chrf_raw,
            'chrfpp_raw': chrfpp_raw,
            'bleu_det':  bleu_det,
            'chrf_det':  chrf_det,
            'chrfpp_det': chrfpp_det,
        })

    n = len(results)
    avg = {k: sum(r[k] for r in results) / n
           for k in ('bleu_raw', 'chrf_raw', 'chrfpp_raw',
                     'bleu_det', 'chrf_det', 'chrfpp_det')}

    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f"English   : {r['english']}\n\n")

            f.write(f"Gold (raw): {r['gold_raw']}\n")
            f.write(f"Pred (raw): {r['pred_raw']}\n")
            f.write(f"BLEU:   {r['bleu_raw']:.4f}\n")
            f.write(f"chrf:   {r['chrf_raw']:.4f}\n")
            f.write(f"chrf++: {r['chrfpp_raw']:.4f}\n\n")

            f.write(f"Gold (det): {r['gold_det']}\n")
            f.write(f"Pred (det): {r['pred_det']}\n")
            f.write(f"BLEU:   {r['bleu_det']:.4f}\n")
            f.write(f"chrf:   {r['chrf_det']:.4f}\n")
            f.write(f"chrf++: {r['chrfpp_det']:.4f}\n")
            f.write("\n" + "-"*60 + "\n\n")

        f.write("=" * 60 + "\n")
        f.write("AVERAGES\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Sentences: {n}\n\n")
        f.write(f"BLEU average of raw:   {avg['bleu_raw']:.4f}\n")
        f.write(f"chrf average of raw:   {avg['chrf_raw']:.4f}\n")
        f.write(f"chrf++ average of raw: {avg['chrfpp_raw']:.4f}\n\n")
        f.write(f"BLEU average of det:   {avg['bleu_det']:.4f}\n")
        f.write(f"chrf average of det:   {avg['chrf_det']:.4f}\n")
        f.write(f"chrf++ average of det: {avg['chrfpp_det']:.4f}\n")

    print(f"\nAverages (raw)  — BLEU: {avg['bleu_raw']:.3f}  chrF: {avg['chrf_raw']:.3f}  chrF++: {avg['chrfpp_raw']:.3f}")
    print(f"Averages (det)  — BLEU: {avg['bleu_det']:.3f}  chrF: {avg['chrf_det']:.3f}  chrF++: {avg['chrfpp_det']:.3f}")
    print("✅ Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Score Navajo MT predictions with sentence BLEU, chrF, chrF++'
    )
    parser.add_argument('--input_file', required=True,
                        help='Predictions file (English/Pred (raw)/Pred (det)/Gold (raw)/Gold (det) format)')
    parser.add_argument('--output_file', default=None,
                        help='Output file (default: adds _scored to input filename)')
    args = parser.parse_args()

    if args.output_file is None:
        p = Path(args.input_file)
        args.output_file = str(p.parent / f"{p.stem}_scored{p.suffix}")

    evaluate(args.input_file, args.output_file)


if __name__ == '__main__':
    main()

