import json
import os
import re

INPUT_DIR = "/home/common/ACNLP/eng2nav/prompt-pipeline/bertscore_bibl_outputs"
OUTPUT_JSON = "/home/common/ACNLP/eng2nav/prompt-pipeline/bertscore_eng_umr.json"

def flatten_umr(umr_text):
    """Remove newlines and extra spaces from UMR while preserving parentheses."""
    # Remove newlines and leading/trailing spaces
    flat = " ".join(line.strip() for line in umr_text.splitlines())
    # Collapse multiple spaces into one
    flat = re.sub(r"\s+", " ", flat)
    return flat

def extract_sentences_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"(?=# ::source)", content)
    sentences = []

    for block in blocks:
        if not block.strip():
            continue
        snt_match = re.search(r"# ::snt (.+)", block)
        if not snt_match:
            continue
        sentence = snt_match.group(1).strip()

        umr_match = re.search(r"# ::date .+\n([\s\S]+)", block)
        umr = flatten_umr(umr_match.group(1)) if umr_match else ""

        sentences.append({"eng": sentence, "eng_umr": umr})

    if not sentences:
        return None, None

    new_sentence = sentences[0]
    five_shot_examples = sentences[1:6]

    return new_sentence, five_shot_examples

all_data = {}
for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.endswith(".txt_output.txt"):
        continue
    fpath = os.path.join(INPUT_DIR, fname)
    new_sentence, five_shot_examples = extract_sentences_from_file(fpath)
    if new_sentence:
        all_data[fname] = {
            "new_sentence": new_sentence,
            "five_shot_examples": five_shot_examples
        }

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

# print one example
example_file = next(iter(all_data))
print("Example from first file:")
print(json.dumps(all_data[example_file], ensure_ascii=False, indent=2))

