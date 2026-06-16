import numpy as np
from scipy.stats import ttest_rel

def load_scores(path):
    scores = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # score lines are numeric
            try:
                scores.append(float(line))
            except ValueError:
                continue
    return np.array(scores)

# Example files
scores_a = load_scores("chrf_base_results/bleu_scores/BLEUSCORESgpt_chrf_nogloss_results_filled.txt") # no morpho
scores_b = load_scores("chrf_final_results/bleu_scores/BLEUSCORESgpt_chrf_withgloss_results_pt2_filled.txt") # with morpho

assert len(scores_a) == len(scores_b), "Score files must align!"

t_stat, p_value = ttest_rel(scores_a, scores_b)

print(f"t-statistic: {t_stat:.4f}")
print(f"two-tailed p-value: {p_value:.6f}")
print(f"mean A: {scores_a.mean():.5f}")
print(f"mean B: {scores_b.mean():.5f}")

