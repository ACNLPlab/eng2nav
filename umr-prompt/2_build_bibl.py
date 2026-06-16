import os
import re

# Paths
BASE = "/home/common/ACNLP/eng2nav/prompt-pipeline"
TEMPLATE_FILE = os.path.join(BASE, "english_template.umr")
SIM_FILE = os.path.join(BASE, "bertscore_most_similar.txt")
OUT_DIR = os.path.join(BASE, "bertscore_bibl_files")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)


def parse_sim_file(sim_file):
    """
    Parse multi-reference similarity file into a dictionary.
    Each entry will store reference + top 5 similar sentences.
    """
    with open(sim_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    blocks = re.split(r"(?=REFERENCE \d+:)", content)
    data = {}

    for block in blocks:
        if not block.strip():
            continue

        # Extract reference number and sentence
        ref_match = re.search(r"REFERENCE (\d+): (.+)$", block, re.MULTILINE)
        if not ref_match:
            continue

        ref_idx = int(ref_match.group(1))
        ref_sentence = ref_match.group(2).strip()
        
        # Extract similar sentences - be more flexible with the pattern
        similar_sents = []
        lines = block.split('\n')
        
        for line in lines:
            # Match lines like: "1. (Score 25.0585) [idx 175] South from the trading post..."
            match = re.match(r'^\d+\.\s*\(Score[^)]+\)\s*\[idx\s+\d+\]\s*(.+)', line)
            if match:
                similar_sents.append(match.group(1).strip())
        
        # Make sure we have at least some similar sentences
        if similar_sents:
            data[ref_idx] = [ref_sentence] + similar_sents[:5]  # Reference + up to 5 similar
        else:
            # If no similar sentences found, just use the reference
            data[ref_idx] = [ref_sentence]

    return data


def fill_template(template_file, sentences):
    """
    Fill the template with the given sentences.
    """
    with open(template_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    sent_idx = 0

    for line in lines:
        if line.strip().startswith("# ::snt"):
            if sent_idx < len(sentences):
                prefix = line.split("# ::snt")[0] + "# ::snt "
                new_lines.append(f"{prefix}{sentences[sent_idx]}\n")
                sent_idx += 1
            else:
                # If we run out of sentences, use the last available one
                prefix = line.split("# ::snt")[0] + "# ::snt "
                new_lines.append(f"{prefix}{sentences[-1]}\n")
        else:
            new_lines.append(line)

    return "".join(new_lines)


def build_bibl_files(sim_file, template_file, out_dir):
    """
    Build BIBL files for each reference in the similarity file.
    """
    data = parse_sim_file(sim_file)

    for ref_idx, sentences in data.items():
        print(f"Processing reference {ref_idx}: {len(sentences)} sentences")
        
        # Make sure we have exactly 6 sentences
        if len(sentences) < 6:
            # For references with fewer than 5 similar sentences, 
            # cycle through the available sentences to fill all slots
            filled_sentences = sentences.copy()
            while len(filled_sentences) < 6:
                # Add sentences from the available ones in order
                filled_sentences.append(sentences[1 + (len(filled_sentences) - 1) % (len(sentences) - 1)])
            sentences = filled_sentences
        
        content = fill_template(template_file, sentences[:6])  # Use first 6 sentences
        out_file = os.path.join(out_dir, f"ref_{ref_idx:03d}_bibl.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Built {out_file} with {len(sentences[:6])} sentences")


if __name__ == "__main__":
    build_bibl_files(SIM_FILE, TEMPLATE_FILE, OUT_DIR)
