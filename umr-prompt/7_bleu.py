import evaluate
import os
import re

# Initialize BLEU metric
bleu = evaluate.load("bleu")

# Input and output directories
input_dir = "/home/common/ACNLP/eng2nav/umr-prompt/bertscore_results"
output_dir = "/home/common/ACNLP/eng2nav/umr-prompt/bertscore_results/bleu_scores"

# Get all input files
input_files = [
    "eng_engumr_nav_output.txt",
    "eng_nav_output.txt",
    "eng_engumr_nav_navumr_output.txt",
    "eng_nav_navumr_output.txt"
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
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse the content
    scores = []
    output_lines = []

    # Split by "File:" to get each entry
    entries = content.split("File: ")[1:]  # Skip first empty split

    for entry in entries:
        lines = entry.strip().split('\n')

        gold = None
        translation = None

        for line in lines:
            if line.startswith("Gold: "):
                gold = line.replace("Gold: ", "").strip()
            elif line.startswith("Translation: "):
                translation = line.replace("Translation: ", "").strip()

        # Compute BLEU score if both gold and translation exist
        if gold and translation:
            result = bleu.compute(
                predictions=[translation],
                references=[[gold]]  # BLEU expects list of lists for references
            )
            score = result["bleu"]
            scores.append(score)

            # Add to output
            output_lines.append(gold)
            output_lines.append(translation)
            output_lines.append(f"{score:.5f}")
            output_lines.append("")  # Empty line between entries

    # Calculate average
    if scores:
        average = (sum(scores) / len(scores)) * 100
        output_lines.append(f"Average: {average:.4f}")

    # Write output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"Processed {input_file}: {len(scores)} pairs, Average BLEU: {average:.5f}")

