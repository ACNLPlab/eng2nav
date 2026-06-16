import json
import re

def parse_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    entries = []
    current = None
    pending_eng = None
    pending_nav = None

    eng_re = re.compile(r"^English:\s*(.*)")
    nav_re = re.compile(r"^Navajo:\s*(.*)")
    morph_re = re.compile(r"^\[.*\]$")
    ref_re = re.compile(r"^Reference:\s*(.*)")

    for line in lines:

        # Start of new block
        if line.strip().startswith("(REFERENCE"):
            if current:
                entries.append(current)
            current = {"reference": None, "triples": []}
            pending_eng = None
            pending_nav = None
            continue

        if current is None:
            continue

        # Skip standalone numbers (1, 2, 3…)
        if re.fullmatch(r"\d+", line.strip()):
            continue

        # English
        m = eng_re.match(line)
        if m:
            pending_eng = m.group(1).strip()
            pending_nav = None
            continue

        # Navajo
        m = nav_re.match(line)
        if m:
            pending_nav = m.group(1).strip()
            continue

        # Morph line
        if morph_re.match(line.strip()):
            morph = line.strip()

            # Build triple
            current["triples"].append((
                pending_eng,
                pending_nav,
                morph
            ))

            # reset
            pending_eng = None
            pending_nav = None
            continue

        # Reference
        m = ref_re.match(line)
        if m:
            current["reference"] = m.group(1).strip()
            continue

    # append final block
    if current:
        entries.append(current)

    # Convert triples into eng1/nav1/morpho1
    final = []
    for block in entries:
        d = {"reference": block["reference"]}
        for i, (en, nv, mo) in enumerate(block["triples"], start=1):
            if en: d[f"eng{i}"] = en
            if nv: d[f"nav{i}"] = nv
            d[f"morpho{i}"] = mo
        final.append(d)

    return final


# Run
data = parse_file("bertscore_withgloss.txt")
fn = "bertscore_withgloss.json"
# Save
with open(fn, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Parsed " + fn + " ", len(data), "reference blocks.")

