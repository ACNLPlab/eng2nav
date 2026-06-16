import evaluate
import os
import re

# Load CHRF metric (CHRf++ is word_order=2)
chrfpp = evaluate.load("chrf")

# Quick test
results = chrfpp.compute(
    predictions=["text"],
    references=[["gold"]],
    word_order=2
)
print(results["score"])

# Input and output directories
input_dir = "/home/common/ACNLP/eng2nav/morpho-prompt/bertscore_base_results"
output_dir = "/home/common/ACNLP/eng2nav/morpho-prompt/bertscore_base_results/chrf++_scores"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

input_files = [
    "gpt_bertscore_nogloss_results_filled.txt"
]

for input_file in input_files:
    input_path = os.path.join(input_dir, input_file)
    output_file = f"CHRFSCORES++_{input_file}"
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        continue

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    scores = []
    output_lines = []

    # Split into blocks separated by blank lines
    entries = re.split(r"\n\s*\n", content.strip())

    for entry in entries:
        gold = None
        translation = None

        for line in entry.splitlines():
            line = line.strip()

            if line.lower().startswith("gold:"):
                gold = line.split(":", 1)[1].strip()
            elif line.lower().startswith("translation:"):
                translation = line.split(":", 1)[1].strip()

        if gold and translation:
            result = chrfpp.compute(
                predictions=[translation],
                references=[[gold]],  # list of lists
                word_order=2          # activate CHRF++
            )
            score = result["score"]
            scores.append(score)

            output_lines.append(f"SCORE: {score:.3f}")
            output_lines.append(f"GOLD: {gold}")
            output_lines.append(f"PRED: {translation}")
            output_lines.append("")

    if scores:
        average = sum(scores) / len(scores)
        output_lines.append(f"AVERAGE_CHRF++: {average:.3f}")
        print(
            f"Processed {input_file}: "
            f"{len(scores)} pairs, Average CHRF++ = {average:.4f}"
        )
    else:
        print(f"Processed {input_file}: 0 valid sentence pairs found")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

print("All files processed!")

