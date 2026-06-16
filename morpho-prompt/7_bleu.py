import evaluate
import os
import re

# Initialize BLEU metric
bleu = evaluate.load("bleu")

# Input and output directories
input_dir = "/home/common/ACNLP/eng2nav/morpho-prompt/bertscore_base_results"
output_dir = "/home/common/ACNLP/eng2nav/morpho-prompt/bertscore_base_results/bleu_scores"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Input files
input_files = [
    "gpt_bertscore_nogloss_results_filled.txt"
]

# Process each file
for input_file in input_files:
    input_path = os.path.join(input_dir, input_file)
    output_file = f"BLEUSCORES{input_file}"
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

        # Compute BLEU score if both gold and translation exist
        if gold and translation:
            result = bleu.compute(
                predictions=[translation],
                references=[[gold]]  # BLEU expects list of lists
            )
            score = result["bleu"]
            scores.append(score)

            # Write per-entry output
            output_lines.append(gold)
            output_lines.append(translation)
            output_lines.append(f"{score:.5f}")
            output_lines.append("")

    # Calculate and append average
    if scores:
        average = (sum(scores) / len(scores)) * 100
        output_lines.append(f"Average: {average:.4f}")
        print(
            f"Processed {input_file}: "
            f"{len(scores)} pairs, Average BLEU: {average:.5f}"
        )
    else:
        print(f"Processed {input_file}: 0 valid sentence pairs found")

    # Write output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

print("All files processed!")

