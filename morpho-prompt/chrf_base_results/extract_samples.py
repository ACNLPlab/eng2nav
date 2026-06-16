import random

INPUT_FILE = "gpt_chrf_withgloss_results.txt"
NUM_SAMPLES = 15


def load_sentence_pairs(filepath):
    pairs = []

    english = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("English:"):
                english = line.replace("English:", "").strip()

            elif line.startswith("Translation:") and english is not None:
                translation = line.replace("Translation:", "").strip()
                pairs.append((english, translation))

                # reset for next chunk
                english = None

    return pairs


def main():
    pairs = load_sentence_pairs(INPUT_FILE)
    print(f"Loaded {len(pairs)} sentence pairs.\n")

    if len(pairs) < NUM_SAMPLES:
        raise ValueError(
            f"Only {len(pairs)} sentence pairs found, cannot sample {NUM_SAMPLES}."
        )

    for i, (english, translation) in enumerate(
        random.sample(pairs, NUM_SAMPLES), 1
    ):
        print(f"{i}.")
        print(f"English: {english}")
        print(f"Translation: {translation}")
        print("-" * 60)


if __name__ == "__main__":
    main()

