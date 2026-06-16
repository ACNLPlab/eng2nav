import evaluate
import os
import re

# Initialize CHRF metric
chrf = evaluate.load("chrf")

# Input and output directories
input_dir = "/home/common/ACNLP/eng2nav/morpho-prompt/bertscore_base_results"
output_dir = "/home/common/ACNLP/eng2nav/morpho-prompt/bertscore_base_results/chrf_scores"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Input files
input_files = [
    "gpt_bertscore_nogloss_results_filled.txt"
]

# Process each file
for input_file in input_files:
    input_path = os.path.join(input_dir, input_file)
    output_file = f"CHRFSCORES{input_file}"
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        continue

    # Read the input file
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

        # Compute CHRF score if both gold and translation exist
        if gold and translation:
            result = chrf.compute(
                predictions=[translation],
                references=[gold]
            )
            score = result["score"]
            scores.append(score)

            # Write per-entry output
            output_lines.append(gold)
            output_lines.append(translation)
            output_lines.append(f"{score:.3f}")
            output_lines.append("")

    # Calculate and append average
    if scores:
        average = sum(scores) / len(scores)
        output_lines.append(f"Average: {average:.3f}")
        print(
            f"Processed {input_file}: "
            f"{len(scores)} pairs, Average CHRF: {average:.4f}"
        )
    else:
        print(f"Processed {input_file}: 0 valid sentence pairs found")

    # Write output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

print("All files processed!")

