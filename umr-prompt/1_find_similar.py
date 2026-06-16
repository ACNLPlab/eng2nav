import evaluate
import os

BASE = "/home/common/ACNLP/eng2nav/umr_data"
ENG_FILE = os.path.join(BASE, "eng_sents.txt")
OUT_DIR = "/home/common/ACNLP/eng2nav/"
TOP_K = 5

def compute_similarity_all(metric_name):
    metric = evaluate.load(metric_name)
    with open(ENG_FILE, "r", encoding="utf-8") as f:
        sents = [line.strip() for line in f if line.strip()]
    n = len(sents)
    print(f"🔹 Loaded {n} sentences for {metric_name} similarity search.")

    outfile = os.path.join(OUT_DIR, f"{metric_name}_most_similar.txt")
    with open(outfile, "w", encoding="utf-8") as f_out:
        for i, ref_sentence in enumerate(sents):
            if metric_name == "bertscore":
                results = metric.compute(
                    predictions=sents,
                    references=[ref_sentence] * n,
                    lang="en"
                )
                scores = results["f1"]
            else:  # chrF
                scores = [
                    metric.compute(predictions=[s], references=[ref_sentence])["score"]
                    for s in sents
                ]

            # Remove self (score at index i)
            ranked = [
                (j, s, sc)
                for j, (s, sc) in enumerate(zip(sents, scores))
                if j != i
            ]
            ranked.sort(key=lambda x: x[2], reverse=True)
            top = ranked[:TOP_K]

            # Write section for this reference sentence
            f_out.write(f"REFERENCE {i+1}: {ref_sentence}\n")
            for rank, (idx, sent, score) in enumerate(top, 1):
                f_out.write(f"{rank}. (Score {score:.4f}) [idx {idx}] {sent}\n")
            f_out.write("\n")

    print(f"✅ Finished computing top {TOP_K} for {metric_name}.")
    print(f"📄 Saved results to {outfile}")

if __name__ == "__main__":
    compute_similarity_all("chrf")
    compute_similarity_all("bertscore")

