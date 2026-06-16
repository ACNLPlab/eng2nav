import evaluate

# Hardcode your reference sentence
REFERENCE_SENTENCE = "And they told us there'd be another flight on the next morning."

# Load chrF metric
chrf = evaluate.load("chrf")

# Read candidate sentences from file
with open("eng_sents.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

num_sentences = len(sentences)
print(f"Loaded {num_sentences} sentences.\n")

# Compute chrF individually for each sentence
scores = []
for sent in sentences:
    result = chrf.compute(predictions=[sent], references=[REFERENCE_SENTENCE])
    scores.append(result["score"])

# Sort sentences by similarity
ranked = sorted(
    [(i, s) for i, s in enumerate(scores)],
    key=lambda x: x[1],
    reverse=True
)

# Output top 6
top_k = 6
outfile_path = "chrf_most_similar.txt"

with open(outfile_path, "w", encoding="utf-8") as f:
    f.write(f"REFERENCE SENTENCE:\n\"{REFERENCE_SENTENCE}\"\n\n")
    f.write(f"Top {top_k} most similar sentences from file:\n\n")
    
    print(f"REFERENCE SENTENCE:\n\"{REFERENCE_SENTENCE}\"\n")
    print(f"Top {top_k} most similar sentences from file:\n")
    
    for rank, (idx, score) in enumerate(ranked[:top_k], 1):
        line = f"{rank}. (Score {score:.4f}) [{idx}] {sentences[idx]}\n"
        print(line, end="")
        f.write(line)

