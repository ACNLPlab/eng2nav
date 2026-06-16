import os
import json
import re

# === Paths ===
umr_files = [
    "/home/common/ACNLP/eng2nav/umr_data/navajo_umr-0001.umr",
    "/home/common/ACNLP/eng2nav/umr_data/navajo_umr-0002.umr",
    "/home/common/ACNLP/eng2nav/umr_data/navajo_umr-0003.umr",
    "/home/common/ACNLP/eng2nav/umr_data/navajo_umr-0004.umr",
    "/home/common/ACNLP/eng2nav/umr_data/navajo_umr-0005.umr",
    #"/home/common/ACNLP/eng2nav/umr_data/navajo_umr-0006.umr",
]

bertscore_path = "/home/common/ACNLP/eng2nav/prompt-pipeline/chrf_eng_umr.json"
output_path = "/home/common/ACNLP/eng2nav/prompt-pipeline/chrf_eng_nav_umr.json"

# === Step 1: Parse all Navajo UMR files ===
def extract_umr_data(file_path):
    """
    Parses each .umr file and returns a list of dicts:
    {
      "eng": English translation,
      "nav": Navajo sentence,
      "nav_umr": UMR graph (linearized)
    }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    entries = []
    # Split by each sentence block
    blocks = content.split("################################################################################")
    for block in blocks:
        if not block.strip():
            continue

        # Extract Translation(English)
        eng_match = re.search(r'Translation\(English\):\s*"?(.+?)"?\s*\n', block)
        nav_match = re.search(r'Words:\s*(.+?)\nMorphemes:', block, re.DOTALL)
        umr_match = re.search(r'# sentence level graph:\n([\s\S]*?)\n# alignment:', block)

        if not (eng_match and nav_match and umr_match):
            continue

        eng = eng_match.group(1).strip().strip('"')
        nav = re.sub(r'\s+', ' ', nav_match.group(1)).strip()
        nav_umr = re.sub(r'\s+', ' ', umr_match.group(1)).strip()

        entries.append({
            "eng": eng,
            "nav": nav,
            "nav_umr": nav_umr
        })

    return entries


# === Step 2: Build lookup dictionary for English → (Navajo, UMR) ===
nav_lookup = {}
for path in umr_files:
    for entry in extract_umr_data(path):
        # Normalize English text for easier matching
        key = entry["eng"].strip().lower().replace('"', '')
        nav_lookup[key] = {
            "nav": entry["nav"],
            "nav_umr": entry["nav_umr"]
        }

print(f"✅ Loaded {len(nav_lookup)} Navajo entries from {len(umr_files)} files.")


# === Step 3: Load bertscore_eng_umr.json ===
with open(bertscore_path, "r", encoding="utf-8") as f:
    bert_data = json.load(f)


# === Step 4: Match and enrich ===
def enrich_with_nav(data):
    for ref_id, entry in data.items():
        # Handle "new_sentence"
        if "new_sentence" in entry:
            eng_text = entry["new_sentence"]["eng"].strip().lower().replace('"', '')
            if eng_text in nav_lookup:
                entry["new_sentence"]["nav"] = nav_lookup[eng_text]["nav"]
                entry["new_sentence"]["nav_umr"] = nav_lookup[eng_text]["nav_umr"]

        # Handle "five_shot_examples"
        if "five_shot_examples" in entry:
            for ex in entry["five_shot_examples"]:
                eng_text = ex["eng"].strip().lower().replace('"', '')
                if eng_text in nav_lookup:
                    ex["nav"] = nav_lookup[eng_text]["nav"]
                    ex["nav_umr"] = nav_lookup[eng_text]["nav_umr"]
    return data


enriched_data = enrich_with_nav(bert_data)


# === Step 5: Save updated dictionary ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(enriched_data, f, ensure_ascii=False, indent=2)

print(f"✅ Enriched data saved to: {output_path}")

