import numpy as np
from scipy.stats import ttest_rel

def load_scores(path):
    scores = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("CHRF:"):
                try:
                    score = float(line.replace("CHRF:", "").strip())
                    scores.append(score)
                except ValueError:
                    raise ValueError(f"Could not parse score from line: {line}")
    return np.array(scores)

# Example files
scores_a = load_scores(
    "bertscore_results/chrf_scores/CHRFSCORESeng_nav_output.txt"
)  # no UMR

scores_b = load_scores(
    "bertscore_results/chrf_scores/CHRFSCORESeng_nav_navumr_output.txt"
)  # with UMR

assert len(scores_a) == len(scores_b), (
    f"Score files must align! "
    f"A={len(scores_a)}, B={len(scores_b)}"
)

t_stat, p_value = ttest_rel(scores_a, scores_b)

print(f"t-statistic: {t_stat:.4f}")
print(f"two-tailed p-value: {p_value:.6f}")
print(f"mean A: {scores_a.mean():.5f}")
print(f"mean B: {scores_b.mean():.5f}")
print(f"mean diff (B - A): {(scores_b - scores_a).mean():.5f}")

